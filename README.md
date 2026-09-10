# 古建筑多视图重建评估工具集

本仓库收录论文《基于三维数字化的建水明清建筑斜栱形制演变研究——兼论古建筑数字档案构建》所使用的图像预处理、图像质量评价与三维网格（Mesh）评价脚本。工具集面向 Instant-NGP、3D Gaussian Splatting（3DGS/gsplat）、GaussianWrapping 与 RealityScan 等方法生成的结果，主要用于比较新视角合成质量、网格结构状态与古建筑构件的可辨识程度。

这些脚本服务于建筑形制研究中的方法选择，不构成通用三维重建基准。网格评价结果应与原始影像及同场景、同视角白模共同解释。

## 脚本说明

| 标准文件名 | 功能 |
| --- | --- |
| `normalize_exposure_wb.py` | 批量统一照片曝光与白平衡，支持 EXIF、参考图和直方图匹配三种模式 |
| `composite_alpha.py` | 将 RGBA 渲染图按标准 Alpha 合成方式叠加至指定背景，输出 RGB 图像 |
| `batch_crop.py` | 按给定百分比批量提取图像中心区域 |
| `eval_metrics4.py` | 统一计算 PSNR、SSIM 与 LPIPS，检查图像模式、尺寸和 ICC 色彩配置 |
| `run_eval_pipeline.py` | 串联 RGBA 背景合成与图像指标评价 |
| `evaluate_meshes.py` | 串行比较 Instant-NGP、GaussianWrapping 与 RealityScan 网格 |
| `evaluate_meshes_parallel.py` | 与串行版指标一致的多进程网格评价版本 |
| `component_clearance_only.py` | 独立计算连通分量间采样近邻距离及其统计量 |

## 环境要求

- Python 3.9–3.11（推荐）
- Windows 10/11 或主流 Linux 发行版
- 图像评价可在 CPU 上运行；使用 CUDA GPU 可显著加快 LPIPS 计算
- 大型网格评价主要消耗系统内存；并行进程会分别加载网格，峰值内存约随 `--workers` 增加
- 生成含中文标签的统计图时，建议安装 Noto Sans CJK SC、思源黑体、微软雅黑或黑体之一；缺少中文字体时脚本会使用英文标签或回退字体

### 安装依赖

创建独立环境：

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若需要使用 NVIDIA GPU，请根据本机 CUDA 与驱动版本，先从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择匹配的 `torch` 和 `torchvision` 安装命令，再安装其余依赖。没有可用 CUDA 时，`eval_metrics4.py` 会从 `cuda` 自动回退至 CPU，也可显式传入 `--device cpu`。

`trimesh` 的曲率计算可能依赖 `rtree`。若 `pip` 安装失败，可使用 Conda 安装：

```bash
conda install -c conda-forge rtree
```

## 推荐目录结构

```text
reconstruction-evaluation/
├── README.md
├── requirements.txt
├── normalize_exposure_wb.py
├── composite_alpha.py
├── batch_crop.py
├── eval_metrics4.py
├── run_eval_pipeline.py
├── evaluate_meshes.py
├── evaluate_meshes_parallel.py
└── component_clearance_only.py
```

## 图像处理与评价

### 1. 曝光与白平衡统一（可选）

```bash
python normalize_exposure_wb.py INPUT_DIR OUTPUT_DIR \
  --mode reference \
  --ref INPUT_DIR/reference.jpg \
  --clamp 4.0
```

支持的模式：

- `reference`：将各图亮度和通道统计量对齐至参考图；未指定 `--ref` 时，自动选择亮度最接近数据集中位数的图像。
- `exif`：依据快门、光圈和 ISO 估计相对曝光；EXIF 信息不足时回退至 `reference`。
- `histogram`：逐通道匹配参考图直方图。

可选参数：

- `--exposure-only`：只调整曝光。
- `--wb-only`：只调整白平衡。
- `--clamp FLOAT`：限制单通道增益，默认 `4.0`。

输出图像统一命名为 `原文件名_norm.jpg`，并生成 `gains_report.csv` 记录各图的曝光及通道增益。

> 论文的图像评价采用未经均光匀色处理的原始图像。该脚本用于探索性对照或其他数据集预处理；若使用其输出，应单独报告处理范围和参数，不应与原始图像评价结果合并。

### 2. RGBA 合成为固定背景 RGB

RealityScan等软件可能输出带 Alpha 通道的渲染图。以下命令将图像合成至白色背景：

```bash
python composite_alpha.py \
  --input ./renders_rgba \
  --output ./renders_rgb \
  --bg 255 255 255
```

也可使用十六进制颜色：

```bash
python composite_alpha.py -i ./renders_rgba -o ./renders_rgb --bg "#ffffff"
```

脚本支持 PNG、JPEG、BMP、TIFF 等常用格式，并保持原文件名输出。

### 3. 批量中心区域裁剪

保留图像中心宽、高各 30% 的区域：

```bash
python batch_crop.py ./images ./images_center30 30
```

等价的命名参数形式：

```bash
python batch_crop.py -i ./images -o ./images_center30 -p 30
```

参数表示宽度和高度各自保留的百分比。因此，`30` 对应的裁剪区域面积约为原图的 9%，并非保留总面积的 30%。脚本只裁剪图像，不计算评价指标。

### 4. PSNR、SSIM 与 LPIPS 评价

```bash
python eval_metrics4.py \
  --gt_dir ./ground_truth \
  --render_dir ./renders_rgb \
  --device cuda \
  --save_per_image ./results/per_image.csv
```

要求：

- 参考图与渲染图按**完全相同的文件名**配对。
- 支持 PNG、JPEG、BMP 和 TIFF。
- 若渲染图尺寸与参考图不同，脚本会将渲染图以双三次插值缩放至参考图尺寸；参考图不变。
- 默认检查 ICC 配置，并尽可能将非 sRGB 图像转换至 sRGB。使用 `--no_icc_convert` 可跳过转换，但仍会检查配置。

指标实现：

- **PSNR**：遵循 Instant-NGP 的 RGB 全像素均方误差定义，数值越高越好。
- **SSIM**：采用 3DGS 的 11×11 高斯窗口、标准差 1.5 和 RGB 分组卷积实现，数值越高越好。
- **LPIPS**：采用官方 `lpips` 包的 AlexNet 网络，并将输入从 `[0,1]` 转换至 `[-1,1]`，数值越低越好。

终端输出各图指标以及均值、最小值和最大值；指定 `--save_per_image` 时另存逐图 CSV。

### 5. 一步完成背景合成与图像评价

`run_eval_pipeline.py` 要求 `composite_alpha.py` 和 `eval_metrics4.py` 位于同一目录。

```bash
python run_eval_pipeline.py \
  --gt_dir ./ground_truth \
  --render_dir ./renders_rgba \
  --bg 255 255 255 \
  --device cuda \
  --save_per_image ./results/per_image.csv
```

默认在临时目录中保存背景合成结果，评价完成后自动删除。需要保留预处理结果时使用：

```bash
python run_eval_pipeline.py \
  --gt_dir ./ground_truth \
  --render_dir ./renders_rgba \
  --preprocessed_dir ./renders_white \
  --save_per_image ./results/per_image.csv
```

除管线自身参数外，其余参数会原样传递给 `eval_metrics4.py`，例如 `--device cpu` 和 `--no_icc_convert`。

## Mesh 评价

### 1. 输入准备

三个输入网格应来自同一场景：

- `--ngp`：Instant-NGP 直接提取的 `.ply` 或 `.obj` 网格；
- `--gw`：GaussianWrapping 生成的网格；
- `--rs`：RealityScan 生成的 `.ply` 或 `.obj` 网格。

脚本读取有效三角面后，执行质心归零，并以各网格原始轴对齐包围盒的对角线归一化至 1。该处理用于相对比较，不会建立真实尺度，也不能替代控制点、人工实测或激光扫描真值。

### 2. 查看包围盒与设置裁剪范围

打印各网格的原始包围盒：

```bash
python evaluate_meshes.py \
  --ngp ./meshes/ngp.ply \
  --gw ./meshes/gw.ply \
  --rs ./meshes/rs.ply \
  --print_bbox
```

以最大连通分量包围盒生成裁剪配置初稿：

```bash
python evaluate_meshes.py \
  --ngp ./meshes/ngp.ply \
  --gw ./meshes/gw.ply \
  --rs ./meshes/rs.ply \
  --suggest_crop ./crop.json
```

`crop.json` 使用各网格自身的原始坐标系：

```json
{
  "ngp": [-0.3, 0.3, -0.2, 0.5, -0.4, 0.4],
  "gw":  [-1.0, 1.0, -0.8, 0.9, -0.5, 0.6],
  "rs":  [-2.0, 2.0, -1.5, 1.7, -1.0, 1.2]
}
```

自动建议仅为初稿。若主体与背景相连，最大连通分量的包围盒可能仍包含背景，使用前应结合原始网格目视检查。

### 3. 串行完整评价

```bash
python evaluate_meshes.py \
  --ngp ./meshes/ngp.ply \
  --gw ./meshes/gw.ply \
  --rs ./meshes/rs.ply \
  --crop ./crop.json \
  --decimate 2000000 \
  --out_dir ./mesh_eval
```

常用选项：

- `--decimate N`：评价前将三个网格分别降至不超过 N 个三角面；未指定时使用原始面数。
- `--no_symmetry`：跳过镜像对称性分析。
- `--no_curvature`：跳过耗时较长的曲率分析。
- `--no_plots`：不生成统计图。
- `--no_render`：不生成同视角白模。
- `--max_render_faces N`：限制白模绘制所使用的三角面数，默认 40,000。
- `--elev1/--azim1`、`--elev2/--azim2`：设置两组统一观察视角。
- `--labels NAME1 NAME2 NAME3`：自定义三种方法的显示名称。

### 4. 并行完整评价

```bash
python evaluate_meshes_parallel.py \
  --ngp ./meshes/ngp.ply \
  --gw ./meshes/gw.ply \
  --rs ./meshes/rs.ply \
  --decimate 2000000 \
  --workers 2 \
  --out_dir ./mesh_eval_parallel
```

并行版与串行版使用相同指标和输出格式。每个 worker 会独立加载一份网格，内存有限时建议使用 `--workers 1` 或 `--workers 2`。

### 5. 评价内容与输出

完整评价包含：

- 基础规模：顶点数、三角面数、原始包围盒对角线；
- 拓扑状态：水密性、边/顶点流形性、非流形边与顶点、退化面、重复面、自相交；
- 连通性与边界：连通分量数、最大分量占比、边界环数；
- 表面形态：相邻面法线夹角、归一化表面积、体积、平均曲率与高斯曲率；
- 分量形态：微小碎片比例、细长分量比例、覆盖 90% 网格面积所需的分量数；
- 探索性指标：连通分量间采样近邻距离；
- 统一视角白模，用于观察构件轮廓、孔隙、薄壁结构、雕饰、粘连与碎裂。

默认输出：

```text
mesh_eval/
├── mesh_quality_report.txt
├── mesh_quality_report.json
├── mesh_comparison.png
├── metric_comparison.png
├── component_distribution.png
├── curvature_distribution.png
└── component_clearance.png
```

曲率分析内部会将超过 300,000 面的网格副本降面后计算；白模绘制也使用独立的低面数副本。这些操作不改变写入报告的主评价网格。

## 独立计算连通分量间采样近邻距离

```bash
python component_clearance_only.py \
  --ngp ./meshes/ngp.ply \
  --gw ./meshes/gw.ply \
  --rs ./meshes/rs.ply \
  --decimate 500000 \
  --workers 2 \
  --out_dir ./clearance_eval
```

算法流程如下：

1. 选择三角面数不少于 30 的连通分量；
2. 每个分量随机选择不超过 100 个三角面，并在各三角面内生成一个采样点；
3. 使用 KD-tree 搜索其他分量的采样点；
4. 以某分量采样点到异分量采样点的最小距离，作为该分量的采样近邻距离；
5. 统计均值、中位数、P90 和非零间距分量占比。

输出包括 `clearance_report.txt`、`clearance_report.json` 和 `component_clearance.png`。

该指标更准确地描述为**连通分量间采样近邻距离**，并非严格的“构件到异构件表面的最近距离”。由于脚本采用随机采样且当前未设置随机种子，不同运行之间可能产生轻微差异；用于正式统计时，建议保持相同的降面规模并进行多次重复计算，同时报告中位数。较大的间距不必然代表网格质量更高，真实构件分离、噪声碎裂和远距离漂浮物都可能抬高数值，必须结合最大连通分量占比、边界环数、碎片指标、白模和原始影像判断。

> 当前版本虽然提供 `--min_faces` 和 `--n_per_comp` 两个命令行参数，但二者尚未传入实际计算函数，运行时仍固定采用 30 面和 100 个采样点。若后续需要调整阈值，应先在代码中完成参数传递并重新验证结果。

## 结果解释注意事项

1. **指标不能单独排序模型优劣。** 最大连通分量占比高可能表示主体完整，也可能表示相邻构件被错误融合；法线夹角或曲率高可能来自真实雕刻，也可能来自重建噪声。
2. **边界环数不等同于孔洞数。** 多分量网格中，各分量的开放外轮廓也会计入；裁剪还会引入人工边界。
3. **体积仅对水密网格具有明确意义。** 开放网格的体积不应用于方法优劣判断。
4. **降面会改变小尺度拓扑。** 小孔、碎片与局部曲率可能在降面后发生变化，比较时应统一降面策略并记录参数。
5. **白模比较不可省略。** 建议在同一场景、相近视角下检查构件轮廓、真实间隙、薄壁结构、雕饰细节、错误粘连和表面碎裂。
6. **本工具集不提供绝对几何精度。** 未使用真值网格时，指标只能描述不同重建结果的相对结构与表面状态。

## 复现建议

- 保存原始影像、测试/训练划分、相机位姿和模型版本；
- 记录软件版本、训练参数、裁剪框、降面目标、评价视角和运行设备；
- 图像评价应使用相同文件名和相同参考坐标；
- 不要在同一统计表中混合原始图像与均光匀色后的图像；
- 将 CSV、JSON、文本报告和白模与论文表格共同归档；
- 大型网格首次运行时可先加入 `--no_curvature --no_render` 检查流程和内存占用。

## 开源许可与引用

本项目采用 [MIT License](LICENSE)。使用、修改或分发代码时，请保留原版权声明和许可声明。

若将本工具集用于其他研究，请在论文或项目中说明所采用的脚本版本、参数和数据处理流程，并引用相关三维重建方法的原始论文。
