# metal-complex-learnability 代码手册

版本：1.0  
日期：2026-10-19  
仓库：https://github.com/jqChen1566/metal-complex-learnability  
许可证：MIT

## 1. 手册用途

本手册说明论文《Mapping the Learnability of Metal-Complex Properties: A Five-Series (3d–5f) Machine Learning Study and Audited Inverse Design》配套代码的安装、数据准备、运行流程、脚本用途与输出位置。手册面向希望复现论文图表、重新训练模型或在此基础上开展新研究的读者。

仓库不附带原始公共数据集。LnQM、tmQM、AcQM 需要从原始来源下载，并按第 4 节的目录结构放置。

## 2. 仓库结构

```
metal-complex-learnability/
  README.md
  LICENSE
  requirements.txt
  code/                 组成模型、分解、稳健性、迁移、逆设计
  gnn/                  图构建、SchNet/DimeNet++/CC-Schnet、SOAP、冻结特征
  data_prep/            极化率、节点级、边级、RDKit、组成基线
  qc_recompute/         ORCA 输入生成、输出解析、多参考锚点
  figures/              论文图件脚本与随包派生数据
  docs/                 数据布局与复现说明
```

## 3. 环境要求

### 3.1 Python 环境

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要依赖：numpy、scipy、scikit-learn、torch、torch-geometric、h5py、ase、dscribe、rdkit、matplotlib、joblib、pandas。

### 3.2 外部程序

- ORCA 6.1.1：用于 `qc_recompute/` 的独立复算与多参考锚点。
- SpookyNet：可选，用于 `gnn/spookynet_features.py`；需要单独获取源代码与预训练权重。
- MACE-OMol：可选，用于 `gnn/mace_omol_features.py`；需要单独安装并准备权重。
- GPU：可选。组成模型的 MLP 与图模型可在 CPU 上运行；GPU 可显著缩短训练时间。

## 4. 数据布局

### 4.1 环境变量

代码通过环境变量定位数据与输出：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `METAL_ML_DATA` | `data` | 属性表、划分文件与数据集 |
| `METAL_ML_RESULTS` | `results` | 分析输出 |
| `METAL_ML_PARTITIONS` | `$METAL_ML_DATA/partitions` | 划分文件 |
| `LNQM_H5` | `$METAL_ML_DATA/lnqm.h5` | LnQM 原始 HDF5 文件 |
| `GRAPH_DIR` | `$METAL_ML_DATA/graphs` | 图文件与图模型输出 |
| `SPOOKYNET_DIR` | 未设置 | SpookyNet 源代码目录 |
| `METAL_ML_RECOMP` | `$METAL_ML_RESULTS/recompute` | 量子化学复算工作目录 |
| `MULTIREF_DIR` | `$METAL_ML_RECOMP/multi_reference` | 多参考锚点输出目录 |

示例：

```bash
export METAL_ML_DATA=/path/to/data
export METAL_ML_RESULTS=/path/to/results
export LNQM_H5=/path/to/data/lnqm.h5
export GRAPH_DIR=/path/to/data/graphs
```

### 4.2 属性表

```
data/
  lnqm.h5
  lnqm_properties_train.json
  lnqm_properties_val.json
  lnqm_properties_test.json
  lnqm_properties_ext.json
  tmqm_properties.json
  tmqm_properties_4d.json
  tmqm_properties_5d.json
  tmqm_properties_5f.json
  tmqm_y.csv
  partitions/
    lnqm_scaffold_12444.json
    tmqm_scaffold_33086.json
```

属性 JSON 的每条记录包含原子序数、坐标、总电荷、未成对电子数、目标值与原始数据索引。训练、验证、测试文件对应固定划分；`partitions/` 中的 JSON 是权威行索引。

### 4.3 图文件

```
data/graphs/
  graphs_lnqm.pt
  graphs_3d.pt
  graphs_4d.pt
  graphs_5d.pt
  graphs_5f.pt
```

每个图文件是张量字典，包含 `z`、`pos`、`y_homo`、`train_rows`、`val_rows`、`test_rows`。数据准备脚本可追加 `y_pol`、`y_hirsh_q`、`y_hirsh_s`、`bo_pairs`、`bo_vals`。坐标单位为埃，能量单位为电子伏特。

## 5. 快速开始

```bash
# 属性提取与对齐
python code/extract_properties.py
python code/extract_tmqm_45d.py
python code/align_acqm.py

# 组成模型与主要分析
python code/train_composition_models.py --models both
python code/decomposition_analysis.py
python code/robustness.py
python code/uncertainty.py
python code/diagnostics.py
python code/model_families.py

# 五系列、迁移与逆设计
python code/five_series.py
python code/transfer.py
python code/design_benchmark.py
python code/design_rebacktest.py

# 图件（随包 data_raw 可直接运行）
python figures/fig1/plot_fig1.py
python figures/fig2_10/plot_fig2.py
```

## 6. 分步说明

### 6.1 属性提取与对齐

`code/extract_properties.py` 从 LnQM 的 HDF5 文件提取十五个目标，写为 `lnqm_properties_ext.json`，并校验索引与原子数对齐。

`code/extract_tmqm_45d.py` 提取 tmQM 的 4d/5d 单金属子集，生成属性表与骨架划分。

`code/align_acqm.py` 对齐 AcQM 的 HDF5 与属性表，生成 5f 属性表与划分。

### 6.2 组成模型

`code/train_composition_models.py` 在 31 维组成描述符上训练随机森林与多层感知机。默认在训练集内做骨架分组交叉验证，测试集只用于最终报告。

`code/decomposition_analysis.py` 生成 Gini 重要性、置换重要性、电荷消融与留一元素结果。

`code/robustness.py` 运行随机森林多种子、MLP 超参数网格。

`code/uncertainty.py` 评估随机森林树间方差与 MLP 集成方差，输出覆盖率与 Spearman 相关。

`code/diagnostics.py` 按电荷、金属、原子数等维度分层误差，并做留一元素与交叉验证的配对检验。

`code/model_families.py` 运行直方图梯度提升与 SOAP 核岭回归，检验结论的模型家族稳健性。

### 6.3 五系列与迁移

`code/five_series.py` 在五个系列上训练组成模型，输出系列基线与留一金属结果。

`code/gini_ci.py`、`code/ablation_series.py`、`code/partition_sensitivity.py` 分别给出电荷重要性置信区间、消融谱与划分种子敏感性。

`code/transfer.py` 运行 5×5 零样本迁移、微调曲线与源规模控制。

### 6.4 图神经网络

```bash
python gnn/build_graphs.py
python gnn/build_series.py
python gnn/train_gnn.py --model schnet --epochs 300
python gnn/train_gnn.py --model dimenet --epochs 300
python gnn/train_gnn.py --model ccschnet --epochs 300
python gnn/soap_krr.py --cache full
python gnn/spookynet_features.py --cache full
python gnn/mace_omol_features.py --cache full
python gnn/decomposition.py
```

`build_graphs.py` 与 `build_series.py` 生成 PyG 图数据。`train_gnn.py` 支持 SchNet、DimeNet++、CC-Schnet 以及节点级、边级任务。`soap_krr.py` 训练 SOAP 描述符加核岭回归。`spookynet_features.py` 与 `mace_omol_features.py` 使用冻结特征加 Ridge 读出。`decomposition.py` 分析几何增益来源。

### 6.5 逆设计

```bash
python code/inverse_design.py
python code/design_benchmark.py
python code/design_rebacktest.py
```

`inverse_design.py` 训练可微代理并实现投影梯度、随机采样、遗传算法与贪心搜索。`design_benchmark.py` 运行 50 次预算匹配基准。`design_rebacktest.py` 用独立随机森林回测候选，并给出相对起点的配对改善。

### 6.6 量子化学复算

```bash
python qc_recompute/sample_lnqm.py
python qc_recompute/gen_orca_inputs.py
# 在 ORCA 中运行生成的输入
python qc_recompute/analyze_orca.py
python qc_recompute/parse_multiref.py
```

`sample_lnqm.py` 抽取 100 个 LnQM 体系。`gen_orca_inputs.py` 生成默认收敛与加严收敛两套单点输入。`analyze_orca.py` 解析输出并与存储标签比较。`parse_multiref.py` 解析 AVAS-CASCI 与 NEVPT2 锚点结果。

### 6.7 图件

```bash
python figures/fig1/plot_fig1.py
python figures/fig2_10/plot_fig2.py
python figures/fig2_10/plot_fig3.py
python figures/fig2_10/plot_fig4.py
python figures/fig2_10/plot_fig5.py
python figures/fig2_10/plot_fig6.py
python figures/fig2_10/plot_fig7.py
python figures/fig2_10/plot_fig8.py
python figures/fig2_10/plot_fig9.py
python figures/fig2_10/plot_fig10.py
python figures/fig2_10/plot_toc.py
python figures/main/plot_main_figure.py
```

图件脚本从各自目录下的 `data_raw/` 读取派生数据，输出写在脚本同级目录。`figures/fig2_10/make_tiff.py` 将 PNG 转为 600 dpi、LZW 压缩、嵌入 sRGB 配置的 TIFF。

## 7. 脚本清单

### 7.1 code/

| 脚本 | 用途 |
|---|---|
| `common.py` | 路径、单位与 31 维组成特征 |
| `extract_properties.py` | LnQM 十五目标提取 |
| `train_composition_models.py` | 随机森林与 MLP 主链 |
| `decomposition_analysis.py` | Gini、置换、消融、留一元素 |
| `robustness.py` | 种子稳定性与 MLP 网格 |
| `uncertainty.py` | 不确定性校准 |
| `diagnostics.py` | 误差分层与配对检验 |
| `model_families.py` | 梯度提升与核岭回归 |
| `five_series.py` | 五系列基线与留一金属 |
| `extract_tmqm_45d.py` | tmQM 4d/5d 提取 |
| `align_acqm.py` | AcQM 对齐与划分 |
| `transfer.py` | 零样本迁移与微调 |
| `gini_ci.py` | 电荷重要性置信区间 |
| `ablation_series.py` | 五系列电荷消融 |
| `partition_sensitivity.py` | 划分种子敏感性 |
| `inverse_design.py` | 可微逆设计框架 |
| `design_benchmark.py` | 设计基准 |
| `design_rebacktest.py` | 独立回测 |
| `overview_figures.py` | 论文总览图 |
| `benchmark_mlp_device.py` | CPU/GPU 训练基准 |

### 7.2 gnn/

| 脚本 | 用途 |
|---|---|
| `build_graphs.py` | 构建 LnQM 图数据 |
| `build_series.py` | 构建 3d/4d/5d/5f 图数据 |
| `train_gnn.py` | SchNet、DimeNet++、CC-Schnet 训练 |
| `soap_krr.py` | SOAP 描述符加核岭回归 |
| `spookynet_features.py` | SpookyNet 冻结特征加 Ridge |
| `mace_omol_features.py` | MACE-OMol 冻结特征加 Ridge |
| `decomposition.py` | 几何增益分解 |

### 7.3 data_prep/

| 脚本 | 用途 |
|---|---|
| `add_polarizability.py` | 向 LnQM 图加入极化率标签 |
| `add_tmqm_polarizability.py` | 向 tmQM 图加入极化率标签 |
| `rf_baselines.py` | 120 维组成随机森林基线 |
| `atom_additive_baseline.py` | 原子加和极化率基线 |
| `add_node_labels.py` | 加入 Hirshfeld 电荷与自旋节点标签 |
| `rdkit_baseline.py` | Morgan 指纹与理化描述符基线 |
| `add_edge_labels.py` | 加入 Mayer 键级边标签 |

### 7.4 qc_recompute/

| 脚本 | 用途 |
|---|---|
| `sample_lnqm.py` | 抽取 100 个复算体系 |
| `gen_orca_inputs.py` | 生成两套 ORCA 单点输入 |
| `analyze_orca.py` | 解析 ORCA 输出并比较存储标签 |
| `parse_multiref.py` | 解析多参考锚点结果 |

### 7.5 figures/

| 脚本 | 用途 |
|---|---|
| `fig1/plot_fig1.py` | 论文 Fig. 1 |
| `main/plot_main_figure.py` | 模型×系列×性质热图组 |
| `fig2_10/plot_fig2.py` | Fig. 2 |
| `fig2_10/plot_fig3.py` | Fig. 3 |
| `fig2_10/plot_fig4.py` | Fig. 4 |
| `fig2_10/plot_fig5.py` | Fig. 5 |
| `fig2_10/plot_fig6.py` | Fig. 6 |
| `fig2_10/plot_fig7.py` | Fig. 7 |
| `fig2_10/plot_fig8.py` | Fig. 8 |
| `fig2_10/plot_fig9.py` | Fig. 9 |
| `fig2_10/plot_fig10.py` | Fig. 10 |
| `fig2_10/plot_toc.py` | 目录图 |
| `fig2_10/make_tiff.py` | PNG 转 TIFF |

## 8. 输出目录

`METAL_ML_RESULTS` 下的目录保留论文分析阶段的标签：

| 目录 | 内容 |
|---|---|
| `E02/` | 组成模型、种子、超参数、校准、诊断 |
| `E02b/` | 决定因素矩阵、消融、留一元素 |
| `E03/` | 五系列基线、迁移、划分敏感性 |
| `E04/` | 逆设计基准与回测 |

图件脚本输出 PNG、PDF 与 SVG；TIFF 由 `make_tiff.py` 生成。

## 9. 复现口径

- 划分：骨架级分组划分；同一系列内所有模型共用同一划分。
- 标准化：均值与标准差只在训练分区计算。
- 随机种子：默认 42；种子稳定性与划分敏感性脚本显式给出其他种子。
- 单位：能量类目标以电子伏特报告，换算常数为 27.2114；偶极矩为德拜，极化率为原子单位，电荷为电子电荷。
- 测试集：只用于最终报告，不参与模型选择。
- 图神经网络：训练预算、隐藏宽度、层数与早停规则由命令行参数给出；论文中的配置见 Supporting Information。

## 10. 常见问题

### 10.1 找不到属性文件

检查 `METAL_ML_DATA`、`LNQM_H5` 与 `METAL_ML_PARTITIONS` 是否指向正确目录；在仓库根目录运行脚本，或显式导出绝对路径。

### 10.2 CUDA 不可用

代码会自动回退到 CPU。GPU 只影响速度，不影响算法。若需强制 CPU，可设置 `CUDA_VISIBLE_DEVICES` 为空。

### 10.3 torch-geometric 导入失败

确认 `torch-geometric` 与 PyTorch 版本匹配；本仓库的图构建使用纯 torch 实现，不依赖 `torch_cluster` 或 `pyg-lib`。

### 10.4 SpookyNet 或 MACE-OMol 导入失败

这两个基线需要单独安装上游包与预训练权重。设置 `SPOOKYNET_DIR`；MACE-OMol 的权重路径按上游包的说明配置。若不使用这两个基线，可跳过对应脚本。

### 10.5 ORCA 输入无法运行

`qc_recompute/gen_orca_inputs.py` 生成的是自包含输入，包含坐标与 `%pal nprocs 1`。请在本机 ORCA 6.1.1 下运行；不同版本的并行设置与收敛关键词可能不同。

### 10.6 图件字体或宽度异常

图件脚本使用 Arial 与数学文本字体配置。若系统缺少 Arial，matplotlib 会回退到其他字体，图面宽度可能变化；投稿前应在目标环境中重新生成并检查。

## 11. 引用与许可

使用本代码请引用论文：

> Mapping the Learnability of Metal-Complex Properties: A Five-Series (3d–5f) Machine Learning Study and Audited Inverse Design

期刊信息与 DOI 在论文发表后补充。代码采用 MIT 许可证，见仓库 `LICENSE`。

## 12. 更新记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-19 | 1.0 | 首版：公开仓库结构、数据布局、运行流程、脚本清单、输出与常见问题 |
