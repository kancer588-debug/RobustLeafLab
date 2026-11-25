# RobustLeafLab

AI3003 课程项目。输入番茄叶片照片，预测健康、早疫或晚疫。主分支默认模型为 **v0.2.0 双向分型多数票**。它在四套既有三类历史回测中判对 **912/1,251 张（72.90%）**，比 v0.1.0 多判对 5 张。新来源上的独立三类表现仍待验证。

## 运行模型

使用 Python 3.11 或更新版本。克隆主分支或下载它的 ZIP，并在仓库根目录运行以下命令。六个复核头已随主分支提供，基础权重继续使用已发布的 `robustleaflab-core.zip`。已有解压好的 `models/` 可以直接复用。 [主分支 ZIP](https://github.com/kancer588-debug/RobustLeafLab/archive/refs/heads/main.zip) · [基础权重 ZIP](https://github.com/kancer588-debug/RobustLeafLab/releases/download/v0.1.0/robustleaflab-core.zip)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\robustleaflab-core.zip" -DestinationPath .
.\.venv\Scripts\python.exe -m robustleaflab predict .\leaf.jpg
.\.venv\Scripts\python.exe -m robustleaflab predict .\leaf.jpg --model v0.1.0
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
| `v0.1.0` | 上一发布版，在 `dino-health` 基础上加 224 像素早疫复核头 |
| `current` | 默认 v0.2.0，增加三种子早疫复核和双向分型多数票 |

默认模型复用 v0.1.0 的基础网络与三个头，再用 seed 42、45、46 的复核头投票。第一轮投票可将原预测健康的图片复核为早疫。第二轮仅复核原已判病害的图片，用不同输出处理早疫改晚疫和晚疫改早疫。至少两票才改判。权重和阈值均取自锁定的实验。 [新增权重及阈值清单](robustleaflab/directional_weights/manifest.json)

## 已锁定的回测与审计

| 数据 | 图数 | 原 AC42 | v0.1.0 | v0.2.0 |
| --- | ---: | ---: | ---: | ---: |
| PlantCity 三类回测 | 872 | 58.83% | 73.39% | 73.85% |
| PlantWild 三类回测 | 102 | 43.14% | 65.69% | 65.69% |
| Imtiaz 三类回测 | 48 | 54.17% | 68.75% | 68.75% |
| Rachin 三类回测 | 229 | 70.31% | 72.93% | 73.36% |
| 四套三类历史回测合计 | 1,251 | 59.47% | 72.50% | **72.90%** |
| 台湾健康与晚疫回测 | 204 | 74.02% | 75.98% | 75.98% |
| New field 外部审计 | 427 | 40.52% | 44.50% | 44.73% |
| PlantDoc 外部审计 | 262 | 62.98% | 75.19% | 74.81% |
| Agri-Vision4 去背景健康图 | 13 | 30.77% | 100.00% | 100.00% |
| PlantSeg 早晚疫复测 | 128 | 57.81% | 71.88% | 71.09% |

表中为图片级准确率。相较 v0.1.0，四套三类回测净增的 5 张均为早疫，晚疫正确数保持 225 张，健康误报保持 29 张。PlantDoc 和 PlantSeg 的准确率分别下降 0.38 和 0.78 个百分点。New field 和 PlantDoc 仅在模型选择锁定后审计。台湾和 PlantSeg 各只有两种真实类别，Agri-Vision4 只检查健康误报。独立实景三类的 80% 目标仍待验证。 [回测记录和混淆矩阵](results/replays.json)

主分支更新前用九套评估的 2,285 张图逐图核对了 `ac42`、`v0.1.0` 和 `current`，输出标签分别与各自的冻结预测一致。

## 基础权重与历史实验归档

| ZIP | 内容 |
| --- | --- |
| `robustleaflab-core.zip` | 两版 CLI 共用的固定基线与 v0.1.0 基础权重 |
| `robustleaflab-backbone-experiments.zip` | ResNet18 和 ConvNeXt-Tiny 对照与续训权重 |
| `robustleaflab-early-source-experiments.zip` | 早疫公开来源替换实验权重 |
| `robustleaflab-expanded-data-experiments.zip` | 训练数据扩容对照权重 |
| `robustleaflab-network-experiments.zip` | 层级分类和相应对照权重 |
| `robustleaflab-experimental-heads.zip` | 分型、局部特征、健康判定、早疫复核和同批次配对等实验分类头 |

实验 ZIP 保留研究用检查点和目录对应关系。归档包含模型张量与必要参数，逐个文件及 ZIP 的 SHA-256 见历史发布清单。 [v0.1.0 历史发布](https://github.com/kancer588-debug/RobustLeafLab/releases/tag/v0.1.0) · [历史发布清单](release-manifest.json)
