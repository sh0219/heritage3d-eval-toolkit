#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一一组照片的曝光与白平衡，减少 3DGS / NeRF 训练伪影。

核心思想
--------
NeRF / 3DGS 假设同一场景点的辐射度在不同视角下保持一致（photometric consistency）。
当相机自动曝光 / 自动白平衡在相邻帧之间跳变时，这种一致性被破坏，
训练时就会产生浮点物（floaters）、颜色斑块、重影等伪影。
本脚本在进入重建前对图像做“全局色调归一化”，尽量让每张图都像用同一套曝光参数拍摄。

三种模式
--------
1) exif      （有 EXIF 元数据时最准）
   用拍摄参数（快门、ISO、光圈）计算相对曝光量，在 *线性* RGB 下乘增益，
   把所有图拉回同一曝光基准。白平衡用全局中值统计估计。

2) reference （无 EXIF 或场景光照恒定时的稳妥选择）
   选一张参考图，把其它图在 Lab/线性空间向参考图的每通道均值对齐
   （类 Reinhard 色彩迁移，只做全局一阶校正，不做局部风格迁移）。

3) histogram （最激进，逐通道直方图匹配，几乎完全抹平色彩差异）
   适合亮度/色偏漂移严重、且不担心轻微改变局部对比度的场景。
   注意：会损失一些自然阴影关系，谨慎使用。

重要细节
--------
- 曝光增益是线性量，必须在 sRGB -> 线性 RGB 之后乘，否则结果不正确。
- 默认只做“全局一阶校正”（每图每通道一个标量增益），不改变构图与局部对比。
- 生成 gains_report.csv，记录每张图的估计增益，便于检查与复现。
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- 色彩空间 --
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB(0-255, float) -> 线性 RGB (0-1)。"""
    x = np.clip(x, 0.0, 255.0) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """线性 RGB (0-1) -> sRGB(0-255, uint8)。"""
    x = np.clip(x, 0.0, 1.0)
    out = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def downscale_for_stats(img: np.ndarray, max_side: int = 256) -> np.ndarray:
    """缩小到 max_side 以内做统计，加快速度。接受 (H,W,3) 浮点/整数数组。"""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    nh, nw = int(round(h * scale)), int(round(w * scale))
    arr = img.astype(np.float64)
    return np.asarray(Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    ).astype(np.float64) / 255.0


# ---------------------------------------------------------------- EXIF 解析 --
EXIF_SHUTTER = 33434      # ExposureTime
EXIF_FNUMBER = 33437      # FNumber
EXIF_ISO = 34855          # ISOSpeedRatings
EXIF_OFFSET = 34665      # Exif IFD pointer


def _as_frac(val):
    """把 EXIF 里的 rational 转成 float。"""
    try:
        if isinstance(val, tuple):
            num, den = val
            return num / den if den else 0.0
        return float(val)
    except Exception:
        return 0.0


def read_exif(path: str):
    """返回 (shutter_s, f_number, iso)，缺失的为 None。"""
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None, None, None
        shutter = _as_frac(exif.get(EXIF_SHUTTER)) or None
        fnum = _as_frac(exif.get(EXIF_FNUMBER)) or None
        iso = None
        if EXIF_ISO in exif:
            iso = float(exif[EXIF_ISO])
        else:
            ifd = exif.get_ifd(EXIF_OFFSET)
            iso = float(ifd[EXIF_ISO]) if EXIF_ISO in ifd else None
        return shutter, fnum, iso
    except Exception:
        return None, None, None


def relative_exposure(shutter, fnum, iso):
    """按针孔相机的曝光方程估算“感光量” t * ISO / N^2。
    返回一个正数，越大表示画面在同样光照下越亮。缺参数时返回 None。"""
    if not shutter or not iso:
        return None
    if not fnum:
        fnum = 1.0
    return shutter * iso / (fnum * fnum)


# ---------------------------------------------------------------- 主流程 ----
def load_linear(path: str):
    """读图 -> sRGB uint8 ndarray -> 线性 float ndarray(0-1)。"""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return srgb_to_linear(np.asarray(img).astype(np.float64))


def build_index(image_dir: str, exts=("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(image_dir, e)))
        paths.extend(glob.glob(os.path.join(image_dir, e.upper())))
    # 稳定排序，保证结果可复现
    return sorted(set(paths))


def robust_luma_mean(lin: np.ndarray):
    """线性空间亮度均值（加权 RGB）。"""
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    return float(np.mean(lum))


def robust_channel_means(lin: np.ndarray):
    """每通道线性均值，裁剪上下各 2% 以抗高光/噪点。"""
    out = []
    for c in range(3):
        v = lin[..., c]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        vc = np.clip(v, lo, hi)
        out.append(float(np.mean(vc)))
    return out


def estimate_exif_gains(paths):
    """基于 EXIF 曝光参数计算每张图相对参考曝光的增益。"""
    expos = []
    for p in paths:
        sh, fn, iso = read_exif(p)
        expos.append(relative_exposure(sh, fn, iso))
    valid = [(i, e) for i, e in enumerate(expos) if e and e > 0]
    if not valid:
        return None
    med = float(np.median([e for _, e in valid]))
    gains = [None] * len(paths)
    for i, e in valid:
        gains[i] = med / e  # 曝光低的补增益，曝光高的压增益
    return gains


def match_histograms(src: np.ndarray, ref: np.ndarray):
    """逐通道把 src(线性 0-1) 的直方图匹配到 ref。返回线性结果。"""
    out = np.empty_like(src)
    for c in range(3):
        s = src[..., c]
        r = ref[..., c]
        # 累积直方图
        sb, si = np.histogram(s, bins=1024, range=(0, 1))
        rb, ri = np.histogram(r, bins=1024, range=(0, 1))
        scdf = np.cumsum(sb).astype(np.float64) / sb.sum()
        rcdf = np.cumsum(rb).astype(np.float64) / rb.sum()
        map_vals = np.interp(scdf, rcdf, ri[1:])
        idx = np.searchsorted(si[1:], s, side="right") - 1
        idx = np.clip(idx, 0, len(map_vals) - 1)
        out[..., c] = map_vals[idx]
    return out


def main():
    ap = argparse.ArgumentParser(description="统一一组照片的曝光与白平衡")
    ap.add_argument("input_dir", help="照片所在目录")
    ap.add_argument("output_dir", help="输出目录（自动创建）")
    ap.add_argument("--mode", choices=["exif", "reference", "histogram"],
                    default="reference", help="校正模式（默认 reference）")
    ap.add_argument("--ref", default=None,
                    help="参考图路径（reference/histogram 模式用；默认取中位数亮度那张）")
    ap.add_argument("--exposure-only", action="store_true",
                    help="只校正曝光，不改白平衡")
    ap.add_argument("--wb-only", action="store_true",
                    help="只校正白平衡，不改曝光")
    ap.add_argument("--clamp", type=float, default=4.0,
                    help="单通道增益上限，防止过度拉伸（默认4.0）")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    paths = build_index(in_dir)
    if not paths:
        print(f"在 {in_dir} 中没有找到 jpg/png/tif 图片")
        sys.exit(1)
    print(f"找到 {len(paths)} 张图片")

    # 1) 读入并转线性
    print("读取图片并转换到线性空间 ...")
    lin_imgs = [load_linear(p) for p in paths]

    # 2) 每张图统计量（线性空间，缩小图做稳健均值）
    stats = [downscale_for_stats(x) for x in lin_imgs]
    chan_means = [robust_channel_means(s) for s in stats]
    luma_means = [robust_luma_mean(x) for x in lin_imgs]

    # 3) 计算增益
    #    per_channel_gains: 每图每通道一个标量（同时校正曝光+白平衡）
    #    expo_gains:        单独的整体曝光增益（EXIF 模式）
    #    wb_gains:          单独的白平衡增益（EXIF 模式）
    per_channel_gains = [np.ones(3)] * len(paths)
    expo_gains = [1.0] * len(paths)
    wb_gains = [np.ones(3)] * len(paths)
    ref_idx = None

    if args.mode == "exif":
        gains = estimate_exif_gains(paths)
        if gains:
            expo_gains = [g if g is not None else 1.0 for g in gains]
            print("使用 EXIF 曝光参数计算增益")
        else:
            print("警告: 缺少 EXIF 曝光信息，回退到 reference 模式")
            args.mode = "reference"

    # 选择参考图（reference / histogram 模式）—— 可能在 EXIF 回退后执行
    if args.mode in ("reference", "histogram"):
        if args.ref:
            if args.ref in paths:
                ref_idx = paths.index(args.ref)
            else:
                print("警告: 参考图不在输入目录中，改用默认")
        if ref_idx is None:
            ref_idx = int(np.argmin(np.abs(np.array(luma_means)
                                           - float(np.median(luma_means)))))
        print(f"参考图: {os.path.basename(paths[ref_idx])}")

    if args.mode == "reference":
        if args.exposure_only:
            # 只对齐亮度（标量曝光）
            tgt_lum = luma_means[ref_idx]
            for i in range(len(paths)):
                if luma_means[i] > 1e-6:
                    expo_gains[i] = tgt_lum / luma_means[i]
        else:
            # 每通道均值对齐参考图 —— 一个标量同时解决曝光与白平衡
            target = np.array(chan_means[ref_idx])
            for i in range(len(paths)):
                if i == ref_idx:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = np.where(np.array(chan_means[i]) > 1e-6,
                                 target / np.array(chan_means[i]), 1.0)
                g = np.clip(g, 1.0 / args.clamp, args.clamp)
                if args.wb_only:
                    # 只校正色偏，保持整体亮度不变
                    g = g / np.mean(g)
                per_channel_gains[i] = g
        print("以每通道均值对齐参考图 (reference 模式)")

    elif args.mode == "exif" and not args.wb_only:
        # EXIF 曝光增益 + 全局中值白平衡
        if not args.exposure_only:
            gmeans = np.array(chan_means)
            target = np.median(gmeans, axis=0)
            for i in range(len(paths)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = np.where(gmeans[i] > 1e-6, target / gmeans[i], 1.0)
                g = np.clip(g, 1.0 / args.clamp, args.clamp)
                g = g / np.mean(g)  # 保持曝光量，只改色温
                wb_gains[i] = g

    # 5) 应用并输出
    ref_img = None
    if args.mode == "histogram":
        ref_img = lin_imgs[ref_idx]
        print(f"直方图匹配到参考图: {os.path.basename(paths[ref_idx])}")

    report = []
    for i, p in enumerate(paths):
        img = lin_imgs[i]
        if args.mode == "histogram":
            img = match_histograms(img, ref_img)
            g_expo = 1.0
            g_wb = np.ones(3)
        else:
            if args.mode == "reference":
                # per_channel_gains 同时包含曝光+白平衡
                g_expo = 1.0
                g_wb = per_channel_gains[i]
            else:  # exif
                g_expo = expo_gains[i]
                g_wb = wb_gains[i]
            if not args.wb_only:
                img = img * g_expo
            if not args.exposure_only:
                img = img * g_wb.reshape(1, 1, 3)
        img = np.clip(img, 0.0, 1.0)
        out = linear_to_srgb(img)
        name = os.path.basename(p)
        stem, ext = os.path.splitext(name)
        out_path = os.path.join(out_dir, f"{stem}_norm.jpg")
        Image.fromarray(out).save(out_path, quality=95, subsampling=0)
        report.append({
            "image": name,
            "exposure_gain": f"{g_expo:.4f}",
            "wb_gain_r": f"{g_wb[0]:.4f}",
            "wb_gain_g": f"{g_wb[1]:.4f}",
            "wb_gain_b": f"{g_wb[2]:.4f}",
        })

    with open(os.path.join(out_dir, "gains_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    print(f"完成: 已输出 {len(paths)} 张到 {out_dir}")
    

if __name__ == "__main__":
    main()
