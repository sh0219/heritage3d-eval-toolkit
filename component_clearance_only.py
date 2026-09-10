#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部件间隙分析（独立轻量版）
==========================
只做"部件间隙分析"这一项指标，跳过全套评估中的曲率/对称/拓扑/法线等耗时项，
用于快速验证"RS 部件分离 vs GW 部件融合"。

对每个"较大独立部件"(连通分量, 面数>=min_faces)在表面采样点,
用 KD-tree 求该部件到最近"异部件"表面的距离, 每部件的最小值即该部件间隙。

输出:
  - 较大独立部件数 / 部件间隙均值/中位/P90 / 非零间隙部件占比
  - component_clearance.png (间隙分布直方图, 三家叠加)
  - clearance_report.txt / .json

用法:
  conda activate mesh_eval
  python component_clearance_only.py \\
      --ngp /path/to/ngp.ply \\
      --gw  /path/to/gw.ply \\
      --rs  /path/to/rs.ply \\
      --decimate 500000 \\
      --out_dir ./clearance_eval

依赖: numpy, open3d, scipy, matplotlib
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def check_deps():
    missing = []
    for mod in ("numpy", "open3d", "scipy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("        请先安装: pip install " + " ".join(missing))
        sys.exit(1)


def load_mesh(path):
    import open3d as o3d
    if not os.path.isfile(path):
        print(f"[ERROR] 文件不存在: {path}")
        sys.exit(1)
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is None or not mesh.has_triangles():
        print(f"[ERROR] 无法读取或没有三角面: {path}")
        sys.exit(1)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    valid = (
        (tris >= 0).all(axis=1)
        & (tris < len(verts)).all(axis=1)
        & (tris[:, 0] != tris[:, 1])
        & (tris[:, 1] != tris[:, 2])
        & (tris[:, 0] != tris[:, 2])
    )
    tris = tris[valid]
    if len(tris) == 0:
        print(f"[ERROR] 有效三角面为 0: {path}")
        sys.exit(1)
    return verts, tris


def decimate_mesh(verts, tris, target_faces):
    if target_faces is None or len(tris) <= target_faces:
        return verts, tris
    import open3d as o3d
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    tm = tm.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    v = np.asarray(tm.vertices, dtype=np.float64)
    t = np.asarray(tm.triangles, dtype=np.int64)
    if len(t) == 0:
        print("[WARN] 降面后无剩余三角面，保留原网格。")
        return verts, tris
    return v, t


def normalize(verts):
    orig_diag = float(np.linalg.norm(np.ptp(verts, axis=0)))
    verts_n = verts - verts.mean(axis=0)
    if orig_diag > 1e-12:
        verts_n = verts_n / orig_diag
    return verts_n, orig_diag


def component_clearance(verts, tris, min_faces=30, max_comps=2000, n_per_comp=100):
    """较大独立部件间隙分析。返回 (stat, gaps)。
    stat 总包含: n_parts_analyzed, n_components_total, single_dominant_body, note。
    若为单主体网格(无可分离部件)，gaps 为 None。"""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    try:
        clusters, n_tri, _ = m.cluster_connected_triangles()
    except Exception as e:
        print(f"[WARN] 连通分量计算失败: {e}")
        return None, None
    clusters = np.asarray(clusters)
    n_tri = np.asarray(n_tri, dtype=np.float64)
    n_comp_total = int(len(n_tri))
    cand = np.where(n_tri >= min_faces)[0]

    # 无 >=min_faces 的部件，或只有一个主导部件 → 单主体/无独立部件
    if len(cand) == 0:
        return {
            "n_parts_analyzed": 0,
            "n_components_total": n_comp_total,
            "single_dominant_body": True,
            "note": "无 >= min_faces 的独立部件",
        }, None
    if len(cand) > max_comps:
        cand = cand[np.argsort(n_tri[cand])[::-1][:max_comps]]
    if len(cand) <= 1:
        return {
            "n_parts_analyzed": int(len(cand)),
            "n_components_total": n_comp_total,
            "single_dominant_body": True,
            "note": "单主体网格，无可分离部件（间隙分析不适用）",
        }, None

    # 每个部件在三角形内部采样 n_per_comp 个点
    pts_list, lab_list = [], []
    for ci in cand:
        tri_idx = np.where(clusters == ci)[0]
        n = min(n_per_comp, len(tri_idx))
        sel = np.random.choice(len(tri_idx), n, replace=False)
        t = tris[tri_idx[sel]]
        v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
        r1 = np.sqrt(np.random.rand(n))
        r2 = np.random.rand(n)
        pts_list.append((1 - r1)[:, None] * v0 + (r1 * (1 - r2))[:, None] * v1 + (r1 * r2)[:, None] * v2)
        lab_list.append(np.full(n, len(pts_list) - 1))
    P = np.vstack(pts_list)
    L = np.concatenate(lab_list)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        k = min(30, len(P))
        dist, idx = tree.query(P, k=k)
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
    except Exception as e:
        print(f"[WARN] KD-tree 失败(需 scipy): {e}")
        return None, None

    min_cross = np.full(len(P), np.inf)
    for j in range(k):
        same = L[idx[:, j]] == L
        min_cross = np.minimum(min_cross, np.where(same, np.inf, dist[:, j]))

    part_min = np.full(len(cand), np.inf)
    np.minimum.at(part_min, L, min_cross)
    gaps = part_min[np.isfinite(part_min)]
    if len(gaps) == 0:
        return {
            "n_parts_analyzed": int(len(cand)),
            "n_components_total": n_comp_total,
            "single_dominant_body": True,
            "note": "部件间无可测间隙（疑似单主体/互相远离）",
        }, None
    eps = 1e-4  # 相对归一化尺度(对角线=1)
    stat = {
        "n_parts_analyzed": int(len(cand)),
        "n_components_total": n_comp_total,
        "single_dominant_body": False,
        "note": "",
        "clearance_mean": float(gaps.mean()),
        "clearance_median": float(np.median(gaps)),
        "clearance_p90": float(np.percentile(gaps, 90)),
        "nonzero_clearance_ratio": float((gaps > eps).mean()),
    }
    return stat, np.asarray(gaps, dtype=np.float64)


def analyze_one(label, path, decimate):
    t0 = time.time()
    print(f"===== {label}: {os.path.basename(path)} =====")
    verts, tris = load_mesh(path)
    orig_faces = len(tris)
    verts, tris = decimate_mesh(verts, tris, decimate)
    if len(tris) < orig_faces:
        print(f"    > 降面 {orig_faces:,} -> {len(tris):,} 面")
    verts_n, orig_diag = normalize(verts)
    stat, gaps = component_clearance(verts_n, tris)
    if stat is None:
        print("    > 部件间隙分析失败")
        print(f"    [done] {time.time()-t0:.1f}s")
        return label, None, None
    if stat.get("single_dominant_body"):
        print(f"    > 单主体网格(独立部件 {stat['n_parts_analyzed']}): {stat.get('note','')}")
    else:
        print(f"    > 较大独立部件数: {stat['n_parts_analyzed']}, "
              f"间隙中位: {stat.get('clearance_median')}, "
              f"非零间隙占比: {stat.get('nonzero_clearance_ratio')}")
    print(f"    [done] {time.time()-t0:.1f}s")
    stat["label"] = label
    stat["file"] = os.path.basename(path)
    stat["orig_bbox_diag"] = orig_diag
    stat["n_faces_after_decimate"] = int(len(tris))
    return label, stat, gaps


def plot_clearance(labels, gaps_list, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#e08a6b", "#7fb0d0", "#9bc4a0"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plotted = False
    for label, gaps, c in zip(labels, gaps_list, colors):
        if gaps is None or len(gaps) == 0:
            continue
        vals = np.asarray(gaps, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) == 0:
            continue
        bins = np.linspace(0, min(float(vals.max()), 0.3), 60)
        ax.hist(vals, bins=bins, alpha=0.5, color=c, label=f"{label} (n={len(vals)})")
        plotted = True
    if plotted:
        ax.set_xlabel("Component clearance to nearest other part (normalized)")
        ax.set_ylabel("Number of parts")
        ax.set_title("Part clearance distribution (separation vs fusion)")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] 部件间隙分布图: {out_png}")


def main():
    check_deps()
    ap = argparse.ArgumentParser(description="部件间隙分析（独立轻量版）")
    ap.add_argument("--ngp", required=True, help="Instant-NGP mesh (.ply/.obj)")
    ap.add_argument("--gw", required=True, help="GaussianWrapping mesh")
    ap.add_argument("--rs", required=True, help="RealityScan mesh")
    ap.add_argument("--decimate", type=int, default=500_000,
                    help="评估前统一降面到 N 面 (默认 500000; None 表示不降)")
    ap.add_argument("--out_dir", default=".", help="输出目录")
    ap.add_argument("--min_faces", type=int, default=30, help="视为'较大部件'的最小面数 (默认 30)")
    ap.add_argument("--n_per_comp", type=int, default=100, help="每部件采样点数 (默认 100)")
    ap.add_argument("--workers", type=int, default=None, help="并行 worker 数 (默认 min(3, CPU核数))")
    ap.add_argument("--labels", nargs=3, default=["Instant-NGP", "GaussianWrapping", "RealityScan"])
    args = ap.parse_args()

    if args.workers is None:
        args.workers = min(3, os.cpu_count() or 1)
    args.workers = max(1, args.workers)
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [(args.labels[i], p, args.decimate) for i, p in enumerate([args.ngp, args.gw, args.rs])]

    results = {}
    gaps_map = {}
    if args.workers == 1:
        for lab, path, dec in jobs:
            lab2, stat, gaps = analyze_one(lab, path, dec)
            results[lab] = stat
            gaps_map[lab] = gaps
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for lab, stat, gaps in pool.starmap(analyze_one, jobs):
                results[lab] = stat
                gaps_map[lab] = gaps

    # 汇总输出
    txt_path = os.path.join(args.out_dir, "clearance_report.txt")
    json_path = os.path.join(args.out_dir, "clearance_report.json")
    lines = []
    lines.append("=" * 70)
    lines.append("部件间隙分析报告 (归一化: 质心归零 + 包围盒对角线=1)")
    lines.append("=" * 70)
    header = f"{'指标':<28}" + "".join(f"{lab:<26}" for lab in args.labels)
    lines.append(header)
    lines.append("-" * 70)

    def row(name, getter, fmt="{:.4f}"):
        vals = []
        for lab in args.labels:
            s = results.get(lab)
            v = getter(s) if s else None
            vals.append("-" if v is None else fmt.format(v))
        lines.append(f"{name:<28}" + "".join(f"{x:<26}" for x in vals))

    row("较大独立部件数", lambda s: s.get("n_parts_analyzed"), "{:d}")
    row("部件间隙均值", lambda s: s.get("clearance_mean"))
    row("部件间隙中位", lambda s: s.get("clearance_median"))
    row("部件间隙P90", lambda s: s.get("clearance_p90"))
    row("非零间隙部件占比", lambda s: s.get("nonzero_clearance_ratio"))
    lines.append("-" * 70)
    # 单主体说明
    for lab in args.labels:
        s = results.get(lab)
        if s and s.get("single_dominant_body"):
            lines.append(f"  注[{lab}]: 单主体网格(独立部件 {s.get('n_parts_analyzed')}, "
                         f"总连通分量 {s.get('n_components_total')}) — {s.get('note','')}")
    lines.append("")
    lines.append("解读: RS 在薄结构/桥阁场景独立部件多、间隙为非零小值 → 单个部件可分辨;")
    lines.append("      GW 独立部件少(薄结构并入主体) → 部件融合/糊团。")
    lines.append("      注意: 采集失败(如指林寺)的碎裂也会产生多部件, 需结合白模判断。")
    lines.append("      实体物体(如柱础)若为完整单体, 无独立部件, 间隙分析显示'单主体'属正常。")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({lab: results.get(lab) for lab in args.labels}, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\n[OK] 报告: {txt_path}")
    print(f"[OK] JSON: {json_path}")

    png_path = os.path.join(args.out_dir, "component_clearance.png")
    plot_clearance(args.labels, [gaps_map.get(lab) for lab in args.labels], png_path)


if __name__ == "__main__":
    main()
