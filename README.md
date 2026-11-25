# RobustLeafLab

AI3003 课程项目。输入番茄叶片照片，预测健康、早疫或晚疫。仓库收录固定对照、当前开发候选、可运行的推理入口与锁定后评估记录。

## 运行模型

使用 Python 3.11 或更新版本。下载 [v0.1.0 发布包](https://github.com/kancer588-debug/RobustLeafLab/releases/tag/v0.1.0) 中的 `robustleaflab-core.zip`，在仓库根目录解压，得到 `models/manifest.json` 和七个权重文件。Windows PowerShell 示例如下。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\robustleaflab-core.zip" -DestinationPath .
.\.venv\Scripts\python.exe -m robustleaflab predict .\leaf.jpg --model current
.\.venv\Scripts\python.exe -m robustleaflab predict .\leaf.jpg --model ac42 --device cpu
```

`leaf.jpg` 换成自己的叶片图片路径。可一次传入多张图片，输出每图一行 JSON。类别顺序是 `healthy`、`early_blight`、`late_blight`。JSON 中的 `decision_scores` 是用于排序的决策分数。

需要 DINOv2 的模型第一次运行会从上游下载并核对固定版本文件。也可用 `python -m robustleaflab fetch-backbone` 提前准备。运行时可通过 `--models-dir` 和 `--cache-dir` 指定模型和缓存目录。 [DINOv2 固定源码版本](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)

## 模型索引

| CLI 名称 | 用途与组成 |
| --- | --- |
| `a42` | 历史 ResNet18 分支 A42，AC42 的父模型 |
| `c42` | 历史 ResNet18 分支 C42，AC42 的父模型 |
| `ac42` | 原 AC42 固定对照，按浮点张量等权合并 A42 和 C42，整数缓冲区沿用 C42 |
| `ac42-finetuned` | 第一次公开数据微调的 ResNet18 |
| `dino-subtype` | 第一次 AC42 微调加 DINOv2 早晚疫分型头 |
| `dino-health` | 上述分型模型加单独的健康与病害判定头 |
| `current` | 当前主候选，在 `dino-health` 基础上加 224 像素早疫复核头 |

`current` 的三个 DINOv2 分类头沿用其各自的 seed 42 开发锁定。健康头估计健康与病害的边界，分型头调整两种病害的顺序，早疫复核头只复核被判为健康且疾病分型倾向早疫的图。

## 已锁定的回测与审计

| 数据 | 图数 | 原 AC42 | 当前主候选 |
| --- | ---: | ---: | ---: |
| PlantCity 三类回测 | 872 | 58.83% | 73.39% |
| PlantWild 三类回测 | 102 | 43.14% | 65.69% |
| Imtiaz 三类回测 | 48 | 54.17% | 68.75% |
| Rachin 三类回测 | 229 | 70.31% | 72.93% |
| 台湾健康与晚疫回测 | 204 | 74.02% | 75.98% |
| New field 外部审计 | 427 | 40.52% | 44.50% |
| PlantDoc 外部审计 | 262 | 62.98% | 75.19% |
| Agri-Vision4 去背景健康图 | 13 | 30.77% | 100.00% |

表中为图片级准确率。台湾包含两种真实类别，模型依然输出三类。Agri-Vision4 小样本来自同一拍摄日期，只检查健康误报。New field 和 PlantDoc 仅在模型选择锁定后审计。独立实景三类的 80% 目标仍待验证。完整分母、Macro-F1 与混淆矩阵见 [回测记录](results/replays.json)。

发布前用现有 PlantCity 回测的 872 张图逐张核对了 `ac42` 和 `current` 的输出标签，两者分别与各自的冻结预测全部一致。

## 发布资产

| ZIP | 内容 |
| --- | --- |
| `robustleaflab-core.zip` | 上述 CLI 所需的固定基线与三阶段候选权重 |
| `robustleaflab-backbone-experiments.zip` | ResNet18 和 ConvNeXt-Tiny 对照与续训权重 |
| `robustleaflab-early-source-experiments.zip` | 早疫公开来源替换实验权重 |
| `robustleaflab-expanded-data-experiments.zip` | 训练数据扩容对照权重 |
| `robustleaflab-network-experiments.zip` | 层级分类和相应对照权重 |
| `robustleaflab-experimental-heads.zip` | 分型、局部特征、健康判定、早疫复核和同批次配对等实验分类头 |

实验 ZIP 保留研究用检查点和目录对应关系。可运行的入口见模型索引。归档包含模型张量与必要参数，逐个文件及 ZIP 的 SHA-256 见 [发布清单](release-manifest.json)。训练图片、测试图片和逐图预测仍保存在项目工作区。
