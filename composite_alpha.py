#!/usr/bin/env python3
"""
RGBA -> 固定背景合成脚本
========================
将输入文件夹中带 alpha 通道的图像（RGBA / LA / PA）按 alpha 值合成到
指定的纯色背景上，并以原文件名（原扩展名）保存到输出文件夹。

对于本身不带 alpha 的图像（RGB / L 等），仅做格式归一化（转成 RGB）
后原样输出，不做合成。

适用场景：
  对 realityscan 等带透明通道的渲染输出做预处理，把它们“压平”到固定
  背景（如白色）后再进行 PSNR / SSIM / LPIPS 等指标评估，避免透明区域
  里未定义的 RGB 像素干扰指标。

运行环境：Python 3.8+  |  Pillow（pip install pillow）

用法：
  python composite_alpha.py -i ./renders -o ./renders_white
  python composite_alpha.py -i ./in -o ./out --bg 255 255 255
  python composite_alpha.py -i ./in -o ./out --bg '#ffffff'
  python composite_alpha.py -i ./in -o ./out --bg 200 210 220
"""

import argparse
from pathlib import Path

from PIL import Image

# 支持的输入扩展名（与 eval_metrics4.py 保持一致）
_COMMON_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def composite_on_background(img: Image.Image, bg_color) -> Image.Image:
    """
    将一张 PIL 图像合成到纯色背景上。

    参数
    ----
    img : PIL.Image.Image
    bg_color : (R, G, B) 三元组，0-255

    返回
    ----
    PIL.Image.Image
        无 alpha 的 RGB 图像。

    说明
    ----
    - 对 RGBA 图像使用 Image.alpha_composite 做标准 alpha 合成，
      公式为 result = fg * alpha + bg * (1 - alpha)。
    - LA / PA 等其它带 alpha 的模式先转为 RGBA 再合成。
    - 无 alpha 的图像直接转 RGB 返回。
    """
    if img.mode in ("RGBA", "LA", "PA"):
        if img.mode == "RGBA":
            rgba = img
        else:
            rgba = img.convert("RGBA")

        # 创建纯色背景图层（与前景同尺寸）
        background = Image.new("RGB", rgba.size, bg_color)
        # alpha_composite 要求两图均为 RGBA 且尺寸一致，先转 RGBA
        bg_rgba = background.convert("RGBA")

        composed = Image.alpha_composite(bg_rgba, rgba)
        return composed.convert("RGB")

    # 无 alpha 通道：仅归一化为 RGB
    return img.convert("RGB")


def _parse_bg_color(tokens) -> tuple:
    """解析 --bg 参数：支持 'R G B' 三个整数，或 '#rrggbb' 十六进制。"""
    if len(tokens) == 3:
        try:
            return tuple(int(t) for t in tokens)
        except ValueError:
            pass
    elif len(tokens) == 1 and str(tokens[0]).startswith("#"):
        h = str(tokens[0]).lstrip("#")
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
    raise argparse.ArgumentTypeError(
        "--bg 需要是 '#rrggbb'（如 #ffffff）或三个整数 'R G B'（如 255 255 255）")


def main():
    parser = argparse.ArgumentParser(
        description="将 RGBA 图像合成到固定背景色，以原文件名输出到指定文件夹。")
    parser.add_argument("-i", "--input", required=True,
                        help="输入文件夹（含待合成的图片）")
    parser.add_argument("-o", "--output", required=True,
                        help="输出文件夹（不存在时自动创建）")
    parser.add_argument("--bg", nargs="+", default=["255", "255", "255"],
                        help="背景颜色，默认白色。支持 '#rrggbb' 或 'R G B'")
    args = parser.parse_args()

    try:
        bg_color = _parse_bg_color(args.bg)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.is_dir():
        parser.error("输入文件夹不存在: {}".format(in_dir))

    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() not in _COMMON_EXTS or not f.is_file():
            continue
        try:
            with Image.open(f) as im:
                mode_before = im.mode
                result = composite_on_background(im, bg_color)
            out_path = out_dir / f.name
            result.save(out_path)
            processed += 1
            print("{:25s}  {} -> {}  saved".format(f.name, mode_before, result.mode))
        except Exception as e:
            print("{:25s}  SKIP: {}".format(f.name, e))
            skipped += 1

    print("\n完成: 处理 {} 张, 跳过 {} 张, 输出到 {}".format(
        processed, skipped, out_dir))


if __name__ == "__main__":
    main()
