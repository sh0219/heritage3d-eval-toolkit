#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图片中心裁剪工具

按照用户输入的百分比裁剪图片，保留图片中心区域，
并以原文件名输出到指定文件夹。

依赖: pip install Pillow (兼容 Python 3.8+)
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("错误: 未找到 Pillow 库，请运行: pip install Pillow")
    sys.exit(1)


def validate_percentage(value: str) -> float:
    """验证并转换百分比参数"""
    try:
        pct = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的百分比值: {value}")

    if pct <= 0 or pct > 100:
        raise argparse.ArgumentTypeError(
            f"百分比必须在 1-100 之间，输入值为: {pct}"
        )
    return pct


def center_crop(image: Image.Image, crop_pct: float) -> Image.Image:
    """
    按百分比对图片中心区域进行裁剪。

    参数:
        image: PIL Image 对象
        crop_pct: 保留的百分比 (1-100)，例如 50 表示保留中心 50% 的区域

    返回:
        裁剪后的 PIL Image 对象
    """
    if crop_pct >= 100:
        return image.copy()

    scale = crop_pct / 100.0
    width, height = image.size

    new_width = int(width * scale)
    new_height = int(height * scale)

    # 计算裁剪边界 (中心裁剪)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def get_image_files(input_dir: str) -> list:
    """获取目录下所有图片文件"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp",
                  ".gif", ".tiff", ".tif", ".webp"}
    image_files = []

    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if os.path.isfile(path) and Path(f).suffix.lower() in extensions:
            image_files.append(f)

    return image_files


def batch_crop(input_dir: str, output_dir: str, crop_pct: float) -> None:
    """
    批量裁剪图片。

    参数:
        input_dir:  输入图片所在的文件夹
        output_dir: 裁剪后图片输出的文件夹
        crop_pct:   保留的百分比 (1-100)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"警告: 在 {input_dir} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，保留比例: {crop_pct}%")
    print("-" * 50)

    success = 0
    skipped = 0
    errors = 0

    for filename in image_files:
        src = input_path / filename
        dst = output_path / filename

        try:
            with Image.open(src) as img:
                cropped = center_crop(img, crop_pct)

                # 保持原图模式 (RGB/RGBA 等)
                if cropped.mode != img.mode and img.mode in ("RGBA", "LA", "P"):
                    cropped = cropped.convert(img.mode)

                cropped.save(dst, quality=95)
            print(f"  ✓ {filename} ({img.size[0]}x{img.size[1]} -> "
                  f"{cropped.size[0]}x{cropped.size[1]})")
            success += 1

        except Exception as e:
            print(f"  ✗ {filename} - 错误: {e}")
            errors += 1

    print("-" * 50)
    print(f"完成: 成功 {success} 张, 失败 {errors} 张")
    print(f"输出目录: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="批量按百分比对图片进行中心裁剪",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_crop.py ./input ./output 50          裁剪至原图的 50%
  python batch_crop.py ./photos ./cropped 80         保留原图中心的 80%
  python batch_crop.py -i ./input -o ./out -p 30     使用命名参数
        """,
    )

    parser.add_argument(
        "input_dir", nargs="?", default=None,
        help="输入图片所在的文件夹路径"
    )
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help="裁剪后图片输出的文件夹路径"
    )
    parser.add_argument(
        "percentage", nargs="?", type=validate_percentage, default=None,
        help="保留的百分比 (1-100)，例如 50 表示保留中心 50%%"
    )
    parser.add_argument(
        "-i", "--input", dest="input_opt",
        help="输入文件夹路径 (命名参数形式)"
    )
    parser.add_argument(
        "-o", "--output", dest="output_opt",
        help="输出文件夹路径 (命名参数形式)"
    )
    parser.add_argument(
        "-p", "--percentage", dest="pct_opt",
        type=validate_percentage,
        help="保留的百分比 1-100 (命名参数形式)"
    )

    args = parser.parse_args()

    # 合并位置参数和命名参数
    input_dir = args.input_opt or args.input_dir
    output_dir = args.output_opt or args.output_dir
    crop_pct = args.pct_opt or args.percentage

    # 交互式输入 (未提供参数时)
    if input_dir is None:
        input_dir = input("请输入图片所在文件夹路径: ").strip().strip('"').strip("'")
    if crop_pct is None:
        while True:
            try:
                raw = input("请输入保留的百分比 (1-100): ").strip()
                crop_pct = validate_percentage(raw)
                break
            except argparse.ArgumentTypeError as e:
                print(f"  {e}，请重新输入。")
    if output_dir is None:
        default_out = os.path.join(input_dir, "cropped")
        out = input(
            f"请输入输出文件夹路径 (直接回车使用 '{default_out}'): "
        ).strip().strip('"').strip("'")
        output_dir = out if out else default_out

    batch_crop(input_dir, output_dir, crop_pct)


if __name__ == "__main__":
    main()
