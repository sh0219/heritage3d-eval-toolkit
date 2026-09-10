#!/usr/bin/env python3
"""
两步批处理：RGBA 背景合成预处理 + 指标评估
============================================
第一步 用 composite_alpha.py 把 render 文件夹里带 alpha 的图片合成到
       固定背景色（默认白色），输出到一个临时/指定文件夹；
第二步 用 eval_metrics4.py 在合成后的 render 与 ground-truth 之间
       计算 PSNR / SSIM / LPIPS。

典型用法：
  python run_eval_pipeline.py --gt_dir ./gt --render_dir ./renders
  python run_eval_pipeline.py --gt_dir ./gt --render_dir ./renders \
      --bg '#ffffff' --preprocessed_dir ./renders_white --device cuda
  python run_eval_pipeline.py --gt_dir ./gt --render_dir ./renders \
      --no_icc_convert --save_per_image per_image.csv

说明：
  - 默认在系统临时目录下创建预处理输出文件夹，用后自动删除；
    也可用 --preprocessed_dir 指定一个保留位置，此时不会删除。
  - 所有无法识别的参数会原样透传给 eval_metrics4.py
    （例如 --device cpu --no_icc_convert --save_per_image xxx.csv）。
  - 若 render 目录里本身没有带 alpha 的图片，预处理只是做一个
    RGB 归一化拷贝，不影响后续评估。
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_script(name: str) -> Path:
    """在脚本同目录下寻找 composite_alpha.py / eval_metrics4.py。"""
    p = _SCRIPT_DIR / name
    if not p.is_file():
        sys.exit("错误：找不到 {}，请确保它与本脚本放在同一目录。".format(name))
    return p


def main():
    parser = argparse.ArgumentParser(
        description="RGBA 背景合成预处理 + eval_metrics4 评估 的批处理脚本",
        add_help=True)
    parser.add_argument("--gt_dir", required=True,
                        help="ground-truth 图片文件夹")
    parser.add_argument("--render_dir", required=True,
                        help="render 图片文件夹（可能含 RGBA）")
    parser.add_argument("--bg", nargs="+", default=["255", "255", "255"],
                        help="合成背景色，默认白色。支持 '#rrggbb' 或 'R G B'")
    parser.add_argument("--preprocessed_dir", default=None,
                        help="预处理输出文件夹；缺省时用临时目录（用后自动删除）")
    parser.add_argument("--keep_preprocessed", action="store_true",
                        help="即使使用临时目录，也保留预处理结果不删除")
    args, passthrough = parser.parse_known_args()

    composite_script = _find_script("composite_alpha.py")
    eval_script = _find_script("eval_metrics4.py")

    # ---------------------------------------------------------------
    # 第 1 步：RGBA -> 固定背景 预处理
    # ---------------------------------------------------------------
    if args.preprocessed_dir:
        pre_dir = Path(args.preprocessed_dir)
    else:
        pre_dir = Path(tempfile.mkdtemp(prefix="eval_pre_"))

    print("=" * 72)
    print("第 1 步 / 2：背景合成预处理")
    print("  输入 : {}".format(Path(args.render_dir).resolve()))
    print("  输出 : {}".format(pre_dir.resolve()))
    print("  背景 : {}".format(" ".join(args.bg)))
    print("=" * 72)

    cmd1 = [sys.executable, str(composite_script),
            "-i", args.render_dir, "-o", str(pre_dir)]
    if args.bg != ["255", "255", "255"]:
        cmd1 += ["--bg"] + args.bg

    ret1 = subprocess.run(cmd1)
    if ret1.returncode != 0:
        if not args.preprocessed_dir and not args.keep_preprocessed:
            shutil.rmtree(pre_dir, ignore_errors=True)
        sys.exit("预处理失败（退出码 {}），已中止。".format(ret1.returncode))

    # ---------------------------------------------------------------
    # 第 2 步：eval_metrics4 评估
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("第 2 步 / 2：PSNR / SSIM / LPIPS 评估")
    print("  render(合成后) : {}".format(pre_dir))
    print("  gt             : {}".format(Path(args.gt_dir).resolve()))
    if passthrough:
        print("  透传参数       : {}".format(" ".join(passthrough)))
    print("=" * 72)

    cmd2 = [sys.executable, str(eval_script),
            "--gt_dir", args.gt_dir,
            "--render_dir", str(pre_dir)] + passthrough

    ret2 = subprocess.run(cmd2)

    # ---------------------------------------------------------------
    # 收尾：默认临时目录模式自动清理
    # ---------------------------------------------------------------
    if not args.preprocessed_dir and not args.keep_preprocessed:
        shutil.rmtree(pre_dir, ignore_errors=True)
        print("\n[已清理临时预处理目录] {}".format(pre_dir))
    elif args.keep_preprocessed and not args.preprocessed_dir:
        print("\n[保留预处理结果] {}".format(pre_dir))

    sys.exit(ret2.returncode)


if __name__ == "__main__":
    main()
