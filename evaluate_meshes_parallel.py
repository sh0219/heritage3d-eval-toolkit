#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三方法 Mesh 质量对比评估脚本 —— 多核并行版
============================================
与 evaluate_meshes.py 功能完全一致（全部指标 + 4 张图），但把三个 mesh 的
评估并行到多个 CPU 核上执行，大幅缩短总耗时。

并行策略:
  - 三个 mesh 的评估彼此独立 → 用 multiprocessing.Pool 并行跑 3 个 worker。
  - worker 内把渲染用的小网格(≤--max_render_faces)降采样好再返回主进程，
    避免把大 mesh 通过进程间通信传回，节省内存。
  - 其余流程(报告汇总、绘图)仍在主进程顺序执行。

用法示例:
    conda activate gaussian_wrapping
    # 用默认并行度(≤3 个 worker)跑
    python evaluate_meshes_parallel.py \\
        --ngp ./ngp_yuhuangge.ply \\
        --gw  ./gw_yuhuangge_post.ply \\
        --rs  ./rs_yuhuangge.ply \\
        --decimate 1000000 \\
        --out_dir ./eval_output

    # 指定并行度(内存紧张时用 --workers 2, 甚至 1)
    python evaluate_meshes_parallel.py --ngp a.ply --gw b.ply --rs c.ply --workers 2 ...

输出(与单核版一致):
    mesh_quality_report.txt / .json, mesh_comparison.png,
    metric_comparison.png, component_distribution.png, curvature_distribution.png

依赖: numpy, open3d, trimesh, matplotlib
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# 依赖检查
# ----------------------------------------------------------------------
def check_deps():
    missing = []
    for mod in ("numpy", "open3d", "trimesh"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("        请先安装: pip install " + " ".join(missing))
        sys.exit(1)
    try:
        import matplotlib
    except ImportError:
        print("[WARN] 未安装 matplotlib，将跳过截图（pip install matplotlib 可补上）。")
        return False
    return True


# ----------------------------------------------------------------------
# 中文字体检测：找到就启用，找不到返回 False（绘图用英文标签兜底）
# ----------------------------------------------------------------------
def _setup_cjk_font():
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        return False
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "Source Han Sans CN",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei",
        "PingFang SC", "AR PL UMing CN", "Droid Sans Fallback",
    ]
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in names:
            matplotlib.rcParams["font.sans-serif"] = [c, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


# ----------------------------------------------------------------------
# 读取 mesh
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 裁剪
# ----------------------------------------------------------------------
def apply_crop(verts, tris, bbox):
    if bbox is None:
        return verts, tris, 0
    lo = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float64)
    hi = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float64)
    centers = verts[tris].mean(axis=1)
    inside = ((centers >= lo) & (centers <= hi)).all(axis=1)
    keep = inside
    n_removed = int((~keep).sum())
    new_tris = tris[keep]
    if len(new_tris) == 0:
        print("[WARN] 裁剪后无剩余三角面，跳过该 mesh 的裁剪。")
        return verts, tris, 0
    used = np.unique(new_tris)
    vmap = np.zeros(len(verts), dtype=np.int64)
    vmap[used] = np.arange(len(used))
    return verts[used], vmap[new_tris], n_removed


# ----------------------------------------------------------------------
# 建议裁剪框
# ----------------------------------------------------------------------
def largest_component_bbox(verts, tris):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    try:
        clusters, n_tri, _ = m.cluster_connected_triangles()
    except Exception as e:
        print(f"[WARN] 连通分量计算失败: {e}")
        return None
    clusters = np.asarray(clusters)
    n_tri = np.asarray(n_tri)
    if len(n_tri) == 0:
        return None
    largest = int(np.argmax(n_tri))
    tri_sel = tris[clusters == largest]
    verts_sel = np.unique(tri_sel)
    v = verts[verts_sel]
    lo, hi = v.min(axis=0), v.max(axis=0)
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])]


# ----------------------------------------------------------------------
# 统一降面
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------------
def normalize(verts):
    orig_diag = float(np.linalg.norm(np.ptp(verts, axis=0)))
    center = verts.mean(axis=0)
    verts_n = verts - center
    if orig_diag > 1e-12:
        verts_n = verts_n / orig_diag
    return verts_n, orig_diag


# ----------------------------------------------------------------------
# 边界环数
# ----------------------------------------------------------------------
def count_boundary_loops(mesh_tm):
    eu = mesh_tm.edges_unique
    inv = mesh_tm.edges_unique_inverse
    cnt = np.bincount(inv, minlength=len(eu))
    be = eu[cnt == 1]
    if len(be) == 0:
        return 0
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in be:
        parent.setdefault(int(a), int(a))
        parent.setdefault(int(b), int(b))
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = set(find(int(v)) for v in parent)
    return len(roots)


# ----------------------------------------------------------------------
# 法线一致性
# ----------------------------------------------------------------------
def normal_consistency(mesh_tm):
    fn = mesh_tm.face_normals
    adj = mesh_tm.face_adjacency
    if len(adj) == 0:
        return None
    cos = np.abs(np.clip(np.sum(fn[adj[:, 0]] * fn[adj[:, 1]], axis=1), -1.0, 1.0))
    ang = np.degrees(np.arccos(cos))
    return {
        "mean_deg": float(np.mean(ang)),
        "median_deg": float(np.median(ang)),
        "p90_deg": float(np.percentile(ang, 90)),
    }


# ----------------------------------------------------------------------
# 连通分量统计
# ----------------------------------------------------------------------
def components(mesh_o3d):
    try:
        tri_clusters, n_tri, area = mesh_o3d.cluster_connected_triangles()
    except Exception:
        return None
    n_tri = np.asarray(n_tri)
    area = np.asarray(area)
    total = n_tri.sum()
    if total == 0:
        return None
    order = np.argsort(n_tri)[::-1]
    return {
        "num_components": int(len(n_tri)),
        "max_component_tri_ratio": float(n_tri[order[0]] / total),
        "max_component_area_ratio": float(area[order[0]] / area.sum()),
        "top5_tri": [int(x) for x in n_tri[order[:5]]],
    }


# ----------------------------------------------------------------------
# 拓扑健康度
# ----------------------------------------------------------------------
def topology(mesh_o3d):
    out = {
        "watertight": bool(mesh_o3d.is_watertight()),
        "edge_manifold": bool(mesh_o3d.is_edge_manifold()),
        "vertex_manifold": bool(mesh_o3d.is_vertex_manifold()),
    }
    try:
        out["non_manifold_edges"] = int(len(mesh_o3d.get_non_manifold_edges()))
    except Exception:
        out["non_manifold_edges"] = -1
    try:
        out["non_manifold_vertices"] = int(len(mesh_o3d.get_non_manifold_vertices()))
    except Exception:
        out["non_manifold_vertices"] = -1
    try:
        out["degenerate_triangles"] = int(len(mesh_o3d.get_degenerate_triangles()))
    except Exception:
        out["degenerate_triangles"] = -1
    try:
        out["duplicated_triangles"] = int(len(mesh_o3d.get_duplicated_triangles()))
    except Exception:
        out["duplicated_triangles"] = -1
    n_faces = len(mesh_o3d.triangles)
    if n_faces <= 1_000_000:
        try:
            out["self_intersecting"] = bool(mesh_o3d.is_self_intersecting())
        except Exception:
            out["self_intersecting"] = None
    else:
        out["self_intersecting"] = None
        out["_note"] = "self-intersection skipped (faces>1M)"
    return out


# ----------------------------------------------------------------------
# 形状描述
# ----------------------------------------------------------------------
def shape(mesh_tm, mesh_o3d, watertight):
    extents = np.asarray(mesh_tm.extents) if hasattr(mesh_tm, "extents") else np.ptp(mesh_tm.vertices, axis=0)
    extents = np.asarray(extents, dtype=np.float64)
    if extents.size == 3 and extents.min() > 1e-12:
        aspect = float(extents.max() / extents.min())
    else:
        aspect = float("nan")
    area = float(mesh_tm.area)
    try:
        vol = float(mesh_o3d.get_volume()) if watertight else None
    except Exception:
        vol = None
    return {
        "bbox_extents": [float(x) for x in extents],
        "aspect_ratio_longest_shortest": aspect,
        "surface_area_normalized": area,
        "volume_normalized": vol,
    }


# ----------------------------------------------------------------------
# 镜像对称偏差
# ----------------------------------------------------------------------
def symmetry(verts, faces, n_samples=150_000):
    import open3d as o3d
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    pcd = tm.sample_points_uniformly(number_of_points=n_samples)
    P = np.asarray(pcd.points)
    center = P.mean(axis=0)
    C = P - center
    cov = np.cov(C, rowvar=False)
    _, v = np.linalg.eigh(cov)
    axes = v.T
    best = None
    for axis in axes:
        d = (C @ axis)[:, None]
        Pm = P - 2.0 * (d * axis)
        pcd_m = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
        dist = pcd_m.compute_point_cloud_distance(pcd)
        cham = float(np.mean(np.asarray(dist)))
        if best is None or cham < best[0]:
            best = (cham, axis)
    return {
        "mirror_symmetry_chamfer": best[0],
        "symmetry_axis": best[1].tolist(),
    }


# ----------------------------------------------------------------------
# 分量形态分析
# ----------------------------------------------------------------------
def component_shape_analysis(mesh_o3d, verts, tris):
    try:
        clusters, n_tri, area = mesh_o3d.cluster_connected_triangles()
    except Exception as e:
        print(f"[WARN] 分量形态分析失败: {e}")
        return None, None
    clusters = np.asarray(clusters)
    n_tri = np.asarray(n_tri, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    total_tri = n_tri.sum()
    if total_tri == 0:
        return None, None
    n_comp = int(len(n_tri))
    small = n_tri < 5
    micro_tri_ratio = float(n_tri[small].sum() / total_tri)
    micro_comp_ratio = float(small.sum() / n_comp)
    cands = np.where(n_tri >= 50)[0]
    if len(cands) > 2000:
        cands = cands[np.argsort(n_tri[cands])[::-1][:2000]]
    aspects = []
    for ci in cands:
        idx = tris[clusters == ci]
        v = verts[np.unique(idx)]
        ext = np.ptp(v, axis=0)
        if ext.min() > 1e-12:
            aspects.append(ext.max() / ext.min())
    thin_ratio = float(np.mean(np.asarray(aspects) > 3.0)) if aspects else 0.0
    order = np.argsort(area)[::-1]
    cum = np.cumsum(area[order]) / area.sum()
    n_for_90 = int(np.searchsorted(cum, 0.9) + 1)
    return {
        "n_components": n_comp,
        "micro_fragment_tri_ratio": micro_tri_ratio,
        "micro_fragment_comp_ratio": micro_comp_ratio,
        "thin_component_ratio": thin_ratio,
        "n_components_for_90pct_area": n_for_90,
    }, np.asarray(area, dtype=np.float64)


# ----------------------------------------------------------------------
# 部件间隙分析：区分"RS 部件分离" vs "GW 部件融合"
#   对每个较大的独立部件(连通分量)采样点，求它到最近"异部件"表面的距离。
# ----------------------------------------------------------------------
def component_clearance(verts, tris, mesh_o3d, min_faces=30, max_comps=2000, n_per_comp=100):
    try:
        clusters, n_tri, _ = mesh_o3d.cluster_connected_triangles()
    except Exception as e:
        print(f"[WARN] 部件间隙分析失败: {e}")
        return None, None
    clusters = np.asarray(clusters)
    n_tri = np.asarray(n_tri, dtype=np.float64)
    n_comp_total = int(len(n_tri))
    cand = np.where(n_tri >= min_faces)[0]

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
        print(f"[WARN] 部件间隙 KD-tree 失败(需 scipy): {e}")
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
    eps = 1e-4
    return {
        "n_parts_analyzed": int(len(cand)),
        "n_components_total": n_comp_total,
        "single_dominant_body": False,
        "note": "",
        "clearance_mean": float(gaps.mean()),
        "clearance_median": float(np.median(gaps)),
        "clearance_p90": float(np.percentile(gaps, 90)),
        "nonzero_clearance_ratio": float((gaps > eps).mean()),
    }, np.asarray(gaps, dtype=np.float64)


# ----------------------------------------------------------------------
# 曲率分布
# ----------------------------------------------------------------------
def curvature_distribution(mesh_tm, verts, tris):
    import trimesh
    CURV_MAX_FACES = 300_000
    try:
        from trimesh.curvature import (
            discrete_gaussian_curvature_measure,
            discrete_mean_curvature_measure,
        )
        if len(tris) > CURV_MAX_FACES:
            import open3d as o3d
            tm = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(tris),
            )
            tm = tm.simplify_quadric_decimation(target_number_of_triangles=CURV_MAX_FACES)
            vc = np.asarray(tm.vertices, dtype=np.float64)
            tc = np.asarray(tm.triangles, dtype=np.int64)
            tm_sub = trimesh.Trimesh(vertices=vc, faces=tc, process=False)
            decimated = True
        else:
            tm_sub = mesh_tm
            decimated = False
        pts = np.asarray(tm_sub.vertices, dtype=np.float64)
        eu = tm_sub.edges_unique
        if len(eu) > 0:
            r = float(np.median(np.linalg.norm(pts[eu[:, 0]] - pts[eu[:, 1]], axis=1))) * 1.5
            r = max(r, 1e-9)
        else:
            r = 1e-6
        g = discrete_gaussian_curvature_measure(tm_sub, pts, r)
        m = discrete_mean_curvature_measure(tm_sub, pts, r)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "rtree" in msg:
            hint = " -> 请安装 rtree: conda install -c conda-forge rtree  或  pip install rtree"
        elif "scipy" in msg:
            hint = " -> 请安装 scipy: conda install scipy  或  pip install scipy"
        print(f"[WARN] 曲率计算失败(可加 --no_curvature 跳过): {msg}{hint}")
        return None, None

    def _clean(a):
        a = np.asarray(a, dtype=np.float64)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    g = _clean(g)
    m = _clean(m)
    ag = np.abs(g)
    am = np.abs(m)
    if len(ag) == 0:
        return None, None
    stat = {
        "curvature_on_decimated": decimated,
        "gauss_abs_mean": float(ag.mean()),
        "gauss_abs_p90": float(np.percentile(ag, 90)),
        "mean_abs_mean": float(am.mean()),
        "mean_abs_median": float(np.median(am)),
        "mean_abs_p90": float(np.percentile(am, 90)),
    }
    return stat, {"gauss": g, "mean": m}


# ----------------------------------------------------------------------
# 并排白模截图
# ----------------------------------------------------------------------
def render_comparison(items, out_png, max_render_faces=40_000, views=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if views is None:
        views = [("Isometric", 25, -60), ("Front", 5, 0)]
    fig = plt.figure(figsize=(6.2 * len(items), 9.2))

    for col, (name, verts, faces, info) in enumerate(items):
        v, f = verts, faces
        if len(f) > max_render_faces:
            import open3d as o3d
            tm = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(v),
                o3d.utility.Vector3iVector(f),
            )
            try:
                tm = tm.simplify_quadric_decimation(target_number_of_triangles=max_render_faces)
                v = np.asarray(tm.vertices)
                f = np.asarray(tm.triangles)
            except Exception:
                pass
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)

        for row, (vname, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, len(items), row * len(items) + col + 1, projection="3d")
            tri = [[v[fi, 0], v[fi, 1], v[fi, 2]] for fi in f]
            try:
                col_rgba = matplotlib.colors.to_rgba("#cfcfcf")
            except (ValueError, TypeError):
                col_rgba = matplotlib.colors.to_rgba("#cfcfcf")
            facecolors = np.tile(np.array(col_rgba, dtype=np.float64)[None, :], (len(tri), 1))
            try:
                pc = Poly3DCollection(
                    tri,
                    facecolors=facecolors,
                    edgecolors="none",
                    shade=True,
                    lightsource=matplotlib.colors.LightSource(azdeg=azim + 90, altdeg=25),
                )
            except ValueError:
                pc = Poly3DCollection(tri, facecolors=facecolors, edgecolors="none")
            ax.add_collection3d(pc)
            lim = 0.72
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"{name}\n({info['n_verts']:,}v / {info['n_faces']:,}f)", fontsize=12)
            elif col == 0:
                ax.text2D(-0.15, 0.5, vname, transform=ax.transAxes, rotation=90, fontsize=12, va="center")

    fig.suptitle("White-model comparison (normalized)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] 截图已保存: {out_png}")


# ----------------------------------------------------------------------
# 统计图表
# ----------------------------------------------------------------------
def plot_metric_comparison(results, out_dir):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    cjk = _setup_cjk_font()

    def L(cn, en):
        return cn if cjk else en

    metrics = [
        (L("最大分量占比", "Max component ratio"),
         lambda r: (r["components"] or {}).get("max_component_tri_ratio"), False),
        (L("微小碎片面数占比", "Micro-fragment tri ratio"),
         lambda r: (r["components_shape"] or {}).get("micro_fragment_tri_ratio"), False),
        (L("细长分量占比", "Thin component ratio"),
         lambda r: (r["components_shape"] or {}).get("thin_component_ratio"), False),
        (L("法线夹角均值(°)", "Mean normal angle (deg)"),
         lambda r: (r["normal_consistency"] or {}).get("mean_deg"), False),
        (L("表面积(归一化)", "Surface area (norm)"),
         lambda r: (r["shape"] or {}).get("surface_area_normalized"), False),
        (L("边界环数", "Boundary loops"), lambda r: r["boundary_loops"], True),
        (L("曲率|平均|均值", "Mean curvature |mean|"),
         lambda r: (r["curvature"] or {}).get("mean_abs_mean"), True),
    ]
    n = len(metrics)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = ["#e08a6b", "#7fb0d0", "#9bc4a0"]
    for ax, (name, getter, log) in zip(axes, metrics):
        vals = []
        for r in results:
            v = getter(r)
            vals.append(0.0 if v is None else float(v))
        ax.bar(labels, vals, color=colors)
        if log and any(v > 0 for v in vals):
            ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=12)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Method comparison (normalized meshes)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "metric_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[OK] 指标对比图: {p}")


def plot_component_distribution(labels, comp_areas, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#e08a6b", "#7fb0d0", "#9bc4a0"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plotted = False
    for label, areas, c in zip(labels, comp_areas, colors):
        if areas is None or len(areas) == 0:
            continue
        areas = np.sort(np.asarray(areas, dtype=np.float64))[::-1]
        xs = np.arange(1, len(areas) + 1)
        ax.loglog(xs, areas, ".", markersize=3, color=c, label=label)
        plotted = True
    if plotted:
        ax.set_xlabel("Component rank (largest → smallest)")
        ax.set_ylabel("Component area (normalized)")
        ax.set_title("Component size distribution (log-log)")
        ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "component_distribution.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[OK] 分量分布图: {p}")


def plot_curvature_distribution(labels, curv_maps, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#e08a6b", "#7fb0d0", "#9bc4a0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    keys = ["gauss", "mean"]
    titles = ["|Gaussian curvature|", "|Mean curvature|"]
    for ax, key, title in zip(axes, keys, titles):
        plotted = False
        for label, cm, c in zip(labels, curv_maps, colors):
            if cm is None:
                continue
            vals = np.abs(np.asarray(cm[key], dtype=np.float64))
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) == 0:
                continue
            hist, bins = np.histogram(np.log10(vals + 1e-12), bins=60)
            ax.plot((bins[:-1] + bins[1:]) / 2, hist, color=c, lw=1.5, label=label)
            plotted = True
        if plotted:
            ax.set_xlabel("log10(|curvature|)")
            ax.set_ylabel("count")
            ax.set_title(title)
            ax.legend(fontsize=8)
    fig.suptitle("Curvature distribution (surface micro-detail)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, "curvature_distribution.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[OK] 曲率分布图: {p}")


def plot_component_clearance(labels, gaps_list, out_dir):
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
    p = os.path.join(out_dir, "component_clearance.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[OK] 部件间隙分布图: {p}")


# ----------------------------------------------------------------------
# 单个 mesh 的评估 worker（供并行池调用）
#   返回 (report, 渲染用verts, 渲染用faces, extras)。
#   渲染网格已在 worker 内降采样到 render_faces，避免大 mesh 走 IPC。
# ----------------------------------------------------------------------
def _analyze_worker(job):
    label, path, do_symmetry, bbox, decimate_faces, do_curvature, render_faces = job
    import open3d as o3d
    import trimesh

    t0 = time.time()
    print(f"\n===== {label}: {os.path.basename(path)} =====")
    verts, tris = load_mesh(path)

    verts, tris, n_removed = apply_crop(verts, tris, bbox)
    if bbox is not None:
        print(f"        > 已按包围盒裁剪，删除 {n_removed} 个三角面，"
              f"剩余 {len(verts):,} 顶点 / {len(tris):,} 面")

    orig_faces = int(len(tris))
    verts, tris = decimate_mesh(verts, tris, decimate_faces)
    decimated = decimate_faces is not None and len(tris) < orig_faces
    if decimated:
        print(f"        > 已降面 {orig_faces:,} -> {len(tris):,} 面")

    verts_n, orig_diag = normalize(verts)

    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_n),
        o3d.utility.Vector3iVector(tris),
    )
    mesh_tm = trimesh.Trimesh(vertices=verts_n, faces=tris, process=False)

    report = {
        "label": label,
        "file": os.path.basename(path),
        "n_vertices": int(len(verts_n)),
        "n_faces": int(len(tris)),
        "orig_bbox_diag": orig_diag,
        "crop_removed_faces": n_removed,
        "crop_bbox": bbox,
        "decimate_target": decimate_faces,
        "decimated": decimated,
        "orig_faces_before_decimate": orig_faces,
    }

    report["boundary_loops"] = count_boundary_loops(mesh_tm)
    report["normal_consistency"] = normal_consistency(mesh_tm)
    report["components"] = components(mesh_o3d)
    report["topology"] = topology(mesh_o3d)
    report["shape"] = shape(mesh_tm, mesh_o3d, report["topology"]["watertight"])
    if do_symmetry:
        try:
            report["symmetry"] = symmetry(verts_n, tris)
        except Exception as e:
            report["symmetry"] = None
            print(f"[WARN] 对称性计算失败: {e}")

    comp_stat, comp_areas = component_shape_analysis(mesh_o3d, verts_n, tris)
    report["components_shape"] = comp_stat

    # 部件间隙分析（区分"RS 部件分离" vs "GW 部件融合"）
    clear_stat, clear_gaps = component_clearance(verts_n, tris, mesh_o3d)
    report["component_clearance"] = clear_stat

    if do_curvature:
        curv_stat, curv_map = curvature_distribution(mesh_tm, verts_n, tris)
    else:
        curv_stat, curv_map = None, None
    report["curvature"] = curv_stat

    # 渲染用的小网格（≤ render_faces），传回主进程拼图
    rv, rf = None, None
    if render_faces and render_faces > 0:
        rv, rf = verts_n, tris
        if len(rf) > render_faces:
            try:
                tm2 = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(rv),
                    o3d.utility.Vector3iVector(rf),
                )
                tm2 = tm2.simplify_quadric_decimation(target_number_of_triangles=render_faces)
                rv = np.asarray(tm2.vertices, dtype=np.float64)
                rf = np.asarray(tm2.triangles, dtype=np.int64)
            except Exception:
                pass

    print(f"[done] {label} {time.time()-t0:.1f}s")
    return report, rv, rf, {"comp_areas": comp_areas, "curvature": curv_map, "gaps": clear_gaps}


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="三方法 Mesh 质量对比评估（多核并行版）")
    ap.add_argument("--ngp", required=True, help="Instant-NGP 生成的 mesh (.ply 或 .obj)")
    ap.add_argument("--gw", required=True, help="GaussianWrapping 生成的 mesh (.ply)")
    ap.add_argument("--rs", required=True, help="RealityScan 生成的 mesh (.ply 或 .obj)")
    ap.add_argument("--out_dir", default=".", help="输出目录 (默认当前目录)")
    ap.add_argument("--max_render_faces", type=int, default=40_000, help="截图渲染用面数上限 (默认 40000)")
    ap.add_argument("--workers", type=int, default=None,
                    help="并行 worker 数 (默认 min(3, CPU核数))。"
                         "注意: 每个 worker 独立加载一份 mesh，峰值内存≈workers×单份，"
                         "32GB 机器建议 2~3，内存紧张用 1(等价串行)。")
    ap.add_argument("--decimate", type=int, default=None, metavar="N_FACES",
                    help="评估前把每个 mesh 统一降面到 N 个三角面 (如 1000000)。"
                         "不传=用原始面数评估。")
    ap.add_argument("--no_symmetry", action="store_true", help="关闭镜像对称性分析")
    ap.add_argument("--no_curvature", action="store_true", help="关闭曲率分布计算(较慢，可跳过)")
    ap.add_argument("--no_plots", action="store_true", help="跳过统计图表输出")
    ap.add_argument("--no_render", action="store_true", help="跳过白模截图输出")
    ap.add_argument("--labels", nargs=3, default=["Instant-NGP", "GaussianWrapping", "RealityScan"],
                    help="三个方法显示名 (默认 Instant-NGP / GaussianWrapping / RealityScan)")
    ap.add_argument("--crop", default=None,
                    help="裁剪包围盒 JSON 文件。格式 {\"ngp\":[...],\"gw\":[...],\"rs\":[...]}，"
                         "坐标用各自 mesh 的原始坐标系；缺哪个 key 就不裁剪哪个。")
    ap.add_argument("--print_bbox", action="store_true",
                    help="只打印三个 mesh 的原始包围盒并退出")
    ap.add_argument("--suggest_crop", default=None, metavar="OUT_JSON",
                    help="为每个 mesh 计算最大连通分量的包围盒并写出建议 crop.json，然后退出。")
    ap.add_argument("--elev1", type=float, default=25, help="第1行视角: 仰角 elevation (默认 25)")
    ap.add_argument("--azim1", type=float, default=-60, help="第1行视角: 方位角 azimuth (默认 -60)")
    ap.add_argument("--elev2", type=float, default=5, help="第2行视角: 仰角 elevation (默认 5)")
    ap.add_argument("--azim2", type=float, default=0, help="第2行视角: 方位角 azimuth (默认 0)")
    args = ap.parse_args()

    check_deps()
    has_mpl = "matplotlib" in sys.modules

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = [("ngp", args.ngp), ("gw", args.gw), ("rs", args.rs)]

    # 默认并行度
    if args.workers is None:
        args.workers = min(3, os.cpu_count() or 1)
    args.workers = max(1, args.workers)
    print(f"[INFO] 并行 worker 数: {args.workers}")

    crop_map = {}
    if args.crop:
        with open(args.crop, encoding="utf-8") as f:
            crop_map = json.load(f)
        print(f"[INFO] 已加载裁剪配置: {sorted(crop_map.keys())}")

    if args.print_bbox:
        print("各 mesh 原始包围盒 [xmin,xmax, ymin,ymax, zmin,zmax]:")
        for key, path in pairs:
            v, _ = load_mesh(path)
            lo, hi = v.min(axis=0), v.max(axis=0)
            print(f"  {key}: [{lo[0]:.4f},{hi[0]:.4f}, {lo[1]:.4f},{hi[1]:.4f}, {lo[2]:.4f},{hi[2]:.4f}]")
        sys.exit(0)

    if args.suggest_crop:
        sugg = {}
        print("计算各 mesh 最大连通分量包围盒（建议裁剪框初稿）...")
        for key, path in pairs:
            v, t = load_mesh(path)
            bb = largest_component_bbox(v, t)
            sugg[key] = bb
            if bb is None:
                print(f"  {key}: 计算失败")
            else:
                print(f"  {key}: [{bb[0]:.4f},{bb[1]:.4f}, {bb[2]:.4f},{bb[3]:.4f}, {bb[4]:.4f},{bb[5]:.4f}]")
        with open(args.suggest_crop, "w", encoding="utf-8") as f:
            json.dump(sugg, f, indent=2, ensure_ascii=False)
        print(f"[OK] 建议裁剪框已写入: {args.suggest_crop}")
        sys.exit(0)

    # 组装任务
    jobs = []
    for i, (key, path) in enumerate(pairs):
        bbox = crop_map.get(key, crop_map.get(args.labels[i]))
        render_faces = None if args.no_render else args.max_render_faces
        jobs.append((args.labels[i], path, not args.no_symmetry, bbox,
                     args.decimate, not args.no_curvature, render_faces))

    results = []
    render_items = []
    extras_list = []

    t_start = time.time()
    if args.workers == 1:
        for j in jobs:
            report, rv, rf, extras = _analyze_worker(j)
            results.append(report)
            extras_list.append(extras)
            if rv is not None:
                render_items.append((report["label"], rv, rf,
                                     {"n_verts": report["n_vertices"], "n_faces": report["n_faces"]}))
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for report, rv, rf, extras in pool.imap(_analyze_worker, jobs):
                results.append(report)
                extras_list.append(extras)
                if rv is not None:
                    render_items.append((report["label"], rv, rf,
                                         {"n_verts": report["n_vertices"], "n_faces": report["n_faces"]}))
    print(f"\n[INFO] 全部评估完成，耗时 {time.time()-t_start:.1f}s")

    # ---------- 汇总输出 ----------
    txt_path = os.path.join(args.out_dir, "mesh_quality_report.txt")
    json_path = os.path.join(args.out_dir, "mesh_quality_report.json")

    lines = []
    lines.append("=" * 78)
    lines.append("三方法 Mesh 质量对比报告 (自动归一化: 质心归零 + 包围盒对角线=1) [多核版]")
    lines.append("=" * 78)
    header = (f"{'指标':<28}" + "".join(f"{r['label']:<26}" for r in results))
    lines.append(header)
    lines.append("-" * 78)

    def row(name, getter, fmt="{:.4f}"):
        vals = []
        for r in results:
            v = getter(r)
            if v is None:
                vals.append("-")
            else:
                try:
                    vals.append(fmt.format(v))
                except (TypeError, ValueError):
                    vals.append(str(v))
        lines.append(f"{name:<28}" + "".join(f"{x:<26}" for x in vals))

    row("顶点数", lambda r: r["n_vertices"], "{:,}")
    row("三角面数", lambda r: r["n_faces"], "{:,}")
    row("边界环数(≈孔洞)", lambda r: r["boundary_loops"], "{:d}")
    row("最大分量占比(面)", lambda r: (r["components"] or {}).get("max_component_tri_ratio"))
    row("最大分量占比(面积)", lambda r: (r["components"] or {}).get("max_component_area_ratio"))
    row("连通分量数", lambda r: (r["components"] or {}).get("num_components"), "{:d}")
    row("水密(Watertight)", lambda r: r["topology"]["watertight"])
    row("边流形", lambda r: r["topology"]["edge_manifold"])
    row("顶点流形", lambda r: r["topology"]["vertex_manifold"])
    row("非流形边数", lambda r: r["topology"]["non_manifold_edges"], "{:d}")
    row("非流形顶点数", lambda r: r["topology"]["non_manifold_vertices"], "{:d}")
    row("退化三角形", lambda r: r["topology"]["degenerate_triangles"], "{:d}")
    row("重复三角形", lambda r: r["topology"]["duplicated_triangles"], "{:d}")
    row("自相交", lambda r: r["topology"]["self_intersecting"])
    row("相邻面法线夹角均值(°)", lambda r: (r["normal_consistency"] or {}).get("mean_deg"))
    row("相邻面法线夹角中位(°)", lambda r: (r["normal_consistency"] or {}).get("median_deg"))
    row("相邻面法线夹角P90(°)", lambda r: (r["normal_consistency"] or {}).get("p90_deg"))
    row("表面积(归一化)", lambda r: (r["shape"] or {}).get("surface_area_normalized"))
    row("体积(归一化)", lambda r: (r["shape"] or {}).get("volume_normalized"))
    row("细长度(长/短轴)", lambda r: (r["shape"] or {}).get("aspect_ratio_longest_shortest"))
    row("镜像对称偏差", lambda r: (r.get("symmetry") or {}).get("mirror_symmetry_chamfer"))
    row("微小碎片面数占比", lambda r: (r["components_shape"] or {}).get("micro_fragment_tri_ratio"))
    row("微小碎片分量占比", lambda r: (r["components_shape"] or {}).get("micro_fragment_comp_ratio"))
    row("细长分量占比", lambda r: (r["components_shape"] or {}).get("thin_component_ratio"))
    row("90%面积需分量数", lambda r: (r["components_shape"] or {}).get("n_components_for_90pct_area"), "{:d}")
    row("曲率|高斯|均值", lambda r: (r["curvature"] or {}).get("gauss_abs_mean"))
    row("曲率|平均|均值", lambda r: (r["curvature"] or {}).get("mean_abs_mean"))
    row("曲率|平均|P90", lambda r: (r["curvature"] or {}).get("mean_abs_p90"))
    row("较大独立部件数", lambda r: (r["component_clearance"] or {}).get("n_parts_analyzed"), "{:d}")
    row("部件间隙均值", lambda r: (r["component_clearance"] or {}).get("clearance_mean"))
    row("部件间隙中位", lambda r: (r["component_clearance"] or {}).get("clearance_median"))
    row("非零间隙部件占比", lambda r: (r["component_clearance"] or {}).get("nonzero_clearance_ratio"))
    lines.append("-" * 78)

    lines.append("")
    lines.append("解读提示:")
    lines.append("  1) 边界环数: 单分量流形网格中 孔洞数 ≈ 边界环数 - 1；多分量时每个分量外轮廓也计入，会偏大。")
    lines.append("  2) 最大分量占比: 越接近 1.0 说明漂浮物/碎块越少，网格越干净。")
    lines.append("  3) 法线夹角越小=表面越光滑；但斜栱/柱础的雕刻棱角处夹角会大，需结合截图权衡'光洁vs细节'。")
    lines.append("  4) 细长度(长/短轴): 斜栱是细长构件，此值应明显>1；柱础偏矮胖，接近1。")
    lines.append("  5) 对称偏差: 沿PCA主轴镜像到自身的平均距离，越小越接近镜像对称（形制相关），仅作参考。")
    lines.append("  6) 体积仅在水密时有意义；RealityScan/MVS 网格常有开口，体积可能无意义。")
    lines.append("  7) 若使用了 --crop 裁剪: 裁切面会给每个 mesh 新增一个边界环(人工切口)，")
    lines.append("     三家裁得一致时该项仍可比，但别把切口当真实孔洞解读。")
    lines.append("  8) 若使用了 --decimate 降面: 指标基于统一降面后的网格，三家面数量级一致，")
    lines.append("     但降面会合并小孔/抹掉碎块，孔洞与分量占比是'降面后'的近似值，解读时注意。")
    lines.append("  9) 微小碎片(面数<5)占比高 → 噪声残片多；细长分量占比高 → 存在真实薄结构(格栅/翘角)。")
    lines.append(" 10) 曲率绝对值越大 → 表面微几何起伏越强(木石纹理/雕花)。区分'真实细节'与'噪声'：")
    lines.append("     低法线夹角+高曲率 = 干净细节；高法线夹角+高曲率 = 噪声。")
    lines.append(" 11) 部件间隙: 较大独立部件到最近异部件的表面距离。RS 在薄结构/桥阁场景独立部件多且")
    lines.append("     有真实非零间隙 → 单个部件可分辨；GW 独立部件少(薄结构并入主体) → 部件融合/糊团。")
    lines.append("     注意: 采集失败(如指林寺)的碎裂也会产生多部件，需结合白模与碎片光滑度共同判断。")
    lines.append("")
    lines.append("结论建议: 综合 最大分量占比(干净)、孔洞数量级(完整)、法线一致性(光洁)、")
    lines.append("          面数/细长度(细节与形态)，再配合并排截图做最终目视判定。")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\n[OK] 文本报告: {txt_path}")
    print(f"[OK] JSON报告: {json_path}")

    if has_mpl:
        if not args.no_render:
            png_path = os.path.join(args.out_dir, "mesh_comparison.png")
            views = [("View 1", args.elev1, args.azim1), ("View 2", args.elev2, args.azim2)]
            render_comparison(render_items, png_path,
                              max_render_faces=args.max_render_faces, views=views)

        if not args.no_plots:
            plot_metric_comparison(results, args.out_dir)
            plot_component_distribution(
                [r["label"] for r in results],
                [e["comp_areas"] for e in extras_list],
                args.out_dir,
            )
            plot_curvature_distribution(
                [r["label"] for r in results],
                [e["curvature"] for e in extras_list],
                args.out_dir,
            )
            plot_component_clearance(
                [r["label"] for r in results],
                [e["gaps"] for e in extras_list],
                args.out_dir,
            )


if __name__ == "__main__":
    main()
