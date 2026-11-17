# DLAN 论文实验复现包

## 目录与任务映射

```text
scripts/
├── README.md
├── REPRODUCE_FINDINGS.md      # 每项结论、每张当前论文图的具体入口
├── requirements.txt
├── environment_snapshot.json
├── experiment_manifest.json  # 每组配置、历史来源、参数和文件 SHA256
├── data/
│   ├── CE/                   # CO2；包含数据和 README
│   ├── SIC/                  # WuhanOD
│   ├── SHP/                  # WuhanHP
│   ├── GVI/                  # GVI，另含月份扩展输入
│   └── historical/           # 单独标注的旧协议记录
├── models/                   # DLAN、GeoAggregator、GCNNWR 与 GNNWR 标准库适配
├── CE.ipynb
├── SIC.ipynb
├── SHP.ipynb
├── GVI.ipynb
├── plot_results.ipynb        # 现有 R 绘图笔记本，未修改
└── outputs/
    ├── reference/            # 原始已完成实验的配置、预测、检查点、审计和训练历史副本
    └── <RUN_ID>/             # 本次核验/重训的结果与待导入 evidence
```

所有路径相对于本目录解析。模型和数据无需再从仓库根目录 `models/`、`share/` 或 `experiments/` 加载；原路径仅作为来源信息保留。各实验按“读取数据 → 模型实验（DLAN、GeoAggregator、GNNWR、GCNNWR）→ 结果总结”组织，实验代码直接写在笔记本中。

## 数据集

| 新名称 | 原案例 | 样本数 | 协变量 | 坐标维度 | 训练 / 验证 / 测试 |
|---|---|---:|---:|---:|---|
| CE | CO2 | 13,000 | 8 | 3 | 9,732 / 1,643 / 1,625 |
| SIC | WuhanOD | 13,807 | 35 | 4 | 9,618 / 2,104 / 2,085 |
| SHP | WuhanHP | 19,599 | 17 | 2 | 15,475 / 2,196 / 1,928 |
| GVI | GVI | 2,122 | 176 | 2 | 1,696 / 213 / 213 |

逐案例说明见 `data/<case>/README.md`。CE 为未来年份预测；SHP 为整小区留出；GVI 为无缓冲的固定空间组划分和冻结特征回归。SIC 训练目标为 log1p(count)，指标使用恢复后的计数。保留全部坐标，CE 使用 GTNNWR，SIC 使用四维 GNNWR；二维 GCNNWR 不适用于 CE/SIC，相关章节明确标注不运行。

## 依赖

Python 3.12；NumPy、pandas、PyTorch、scikit-learn、JupyterLab、ipykernel、nbformat、nbclient，以及作者发布的 `gnnwr==0.1.17`。先在独立环境中安装与 CUDA/驱动兼容的 PyTorch，再安装其余依赖。

```bash
cd article/scripts
python -m pip install -r requirements.txt
python -m ipykernel install --user --name article-dlan --display-name "Python (DLAN article)"
jupyter lab .
```

若作者包的依赖约束与已验证的 PyTorch 环境冲突，可先安装其余依赖，再执行 `python -m pip install --no-deps gnnwr==0.1.17`。本适配只使用该标准库的 `networks.py`，不调用它的训练器、绘图或诊断模块。网络源码未重复打包；来源及一致性校验见 `models/PROVENANCE.json`。

当前 R 绘图笔记本加载 tidyverse（含 ggplot2/dplyr/tidyr）、patchwork、sf、tmap、ggpubr 和 grid，并使用 viridisLite/scales；还需要 IRkernel。tmap 使用第 4 版 API；sf 需要可用的 GDAL/GEOS/PROJ。原统计绘图环境记录见 `../evidence/R_sessionInfo.txt`，当前绘图额外依赖以笔记本首部为准。已有 R 内核可直接运行 `plot_results.ipynb`。复现架构图另外需要 Pillow、Draw.io Desktop CLI、Inkscape，具体步骤见 `REPRODUCE_FINDINGS.md`。无需训练图像编码器，也不依赖 TensorFlow。

## 默认：核验已有结果并重建证据

依次用新的 Python 内核运行 `CE.ipynb`、`SIC.ipynb`、`SHP.ipynb`、`GVI.ipynb`，或在本目录执行：

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan CE.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan SIC.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan SHP.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan GVI.ipynb
```

默认 `ARTICLE_MODE=replay`，逐个检查 SHA256、行标识、划分、参考库、检查点和已有重载审计，再从保存的预测重新计算验证/测试 RMSE、MAE、R²、尾部误差、社区误差和尺度参数。它不会重新训练，也不把“重新计算指标”称为“重新训练复现”。

最后完成的案例将四个案例的片段汇总到 `outputs/replay/evidence/`。其中包括 `plot_results.ipynb` 当前读取的全部 8 个输入 CSV，以及主表、补充比较与 SIC 3×3 敏感性汇总。核验与重训分别使用独立 RUN_ID；不同模式、不同源清单或种子集合的片段不能混合。

## 从头训练

在启动新内核前设置：

```bash
export ARTICLE_MODE=train
export ARTICLE_RUN_ID=retrain-v1
```

然后按同样顺序执行四个笔记本。该模式训练主比较及被汇总图表需要的历史/补充配置，计算量大；默认每组 10 个种子。SIC 的九格因子实验保持预测头 `(10,8)`，不能与旧的宽度 10、预测头 `(36,3)` 配置混淆。每次拟合保存配置、训练历史、预测、检查点与重载审计，全部审计通过后才写入完成指标。

统一训练设置为 AdamW、学习率 0.001、权重衰减 0.0001、有效 batch 1,024、查询微 batch 64、最多 300 轮、验证损失早停 patience 30、min_delta 1e-5、梯度裁剪 5，ReduceLROnPlateau 的 factor 0.5、patience 10。模型参数预算与候选数按清单逐配置设置。主比较 SIC 为 7,069 参数、DLAN K512；其余案例为 6,957 参数。

GPU 推荐用于完整训练；默认核验可在 CPU 上完成。训练软件、随机数、硬件和原始运行计时均有记录；SIC K512/K2048 的原始实验运行环境不同，不能把记录时间当成同机速度比较。新增拟合将记录自己的计时，不复用原计时冒充新测量。

## 将结果导入论文 evidence

默认 `ARTICLE_PUBLISH_EVIDENCE=0`，不会覆盖 `../evidence/`。确认 `outputs/<RUN_ID>/evidence/` 后，在最后一个案例的准备 cell 设置 `PUBLISH_EVIDENCE=True`，只重跑结果总结末尾的导入 cell；也可在启动内核前设置 `ARTICLE_PUBLISH_EVIDENCE=1` 后重跑。覆盖前的证据会备份到该运行目录。

导入不会执行或编辑 `plot_results.ipynb`，其已有输出保持原样。需要更新图时再显式执行绘图笔记本。每张当前论文图和每项结果的步骤见 `REPRODUCE_FINDINGS.md`。

## 可复现范围

本包包含 27 个独立配置、270 次已完成拟合的冻结产物；14 个主配置对应 140 次拟合。重复引用的 K512、容量或因子实验通过来源路径去重，主比较与历史比较仍分开汇总。历史 Gaussian-prior CSV 单独保留，不声称它是最终 DLAN 的模块消融或由这四个笔记本重新拟合。

模型整理保留原数值操作和 state-dict 键，文件名与导入路径仅作局部调整。标准库 GNNWR 采用固定训练参考点，是预算约束下的作者网络适配，不是作者默认全参考设置的完整复刻。

本次整理不修改 `main.tex`、现有 `article/evidence/`、图片或 `plot_results.ipynb` 的任何内容。

## 当前绘图笔记本的已知执行问题（未修改）

在空 R 内核中用重建证据运行副本，发现以下既有问题：

| Cell（从 1 开始计数） | 问题 | 影响 |
|---|---|---|
| 12 | `mutate()` 引用尚未生成的 `accuracy_long$metric` | 主比较图的绘图表达式失败；其后 `ggsave()` 可能保存此前的图，不能只凭文件存在判定成功。 |
| 16 | 使用尚未定义的 `p` | 计算时间比较导出失败。 |
| 31 | `glimpse(od)` 出现在创建 `od` 的 cell 前 | 从空内核运行时失败；后续 cell 才创建该变量。 |

这三处属于现有绘图代码的变量引用/执行顺序问题，不是输入 CSV 的指标差异。本次按要求没有修改 `plot_results.ipynb` 或它的结果。`outputs/replay/plot_compatibility.json` 记录了执行副本的诊断；不将“部分图已导出”描述为整个绘图笔记本已通过。

SIC 九格敏感性结果的标准差均由该格十个种子重新计算。当前 LaTeX 中部分宽度 10/16 行的标准差与重新计算结果不同；重建结果见 `outputs/replay/evidence/sic_factorial_summary.csv`，本次未改论文表格。

原有 `prepare_evidence.py`、`prepare_supplement_evidence.py` 和 `prepare_reported_results.py` 作为历史入口保留，仍服务旧仓库布局；新复现流程不调用这些脚本。`render_references.py` 属于参考文献检查，也不进入模型实验。
