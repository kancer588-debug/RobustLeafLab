# RobustLeafLab

AI3003 课程项目。输入番茄叶片照片，预测健康、早疫或晚疫。默认模型为 **v0.2.0 双向分型多数票**。在四套既有三类历史回测中判对 **912/1,251 张（72.90%）**，比 v0.1.0 净多判对 **5 张**。同期外部审计中，New field 从 44.50% 到 44.73%，PlantDoc 从 75.19% 到 74.81%。新采集组上的独立三类测试仍待完成。

## 运行模型

在仓库根目录运行。以下是 Windows PowerShell 下的 CPU 验收流程，使用 Python 3.11。下载 [基础权重 ZIP](https://github.com/kancer588-debug/RobustLeafLab/releases/download/v0.1.0/robustleaflab-core.zip) 到下载目录，再安装依赖和解压权重。已有解压好的 `models/` 可以复用。

```powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0
& $python -m pip install Pillow==12.3.0 numpy==2.4.6
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\robustleaflab-core.zip" -DestinationPath .
& $python -m robustleaflab predict examples/images/healthy-148.JPG --model ac42 --device cpu
& $python -m robustleaflab evaluate examples/manifest.csv --model ac42 --device cpu --check-frozen
```

示例单图应输出 `label: healthy`。九图验收集中 AC42 判对 6 张，`frozen_mismatches` 应为 0。验收集是 PlantCity 回测的子集，适合核对运行与输出。

演示默认模型前，先联网准备固定版本的 DINOv2 文件。缓存保存于 `.model-cache/`，文件通过 SHA-256 后可以离线复用。

```powershell
& $python -m robustleaflab fetch-backbone
& $python -m robustleaflab predict examples/images/early_blight-105.JPG --model current --device cpu
& $python -m robustleaflab evaluate examples/manifest.csv --model current --device cpu --check-frozen
```

`105.JPG` 的期望结果为 `early_blight`，默认模型的九图验收为 7 张正确和 0 条冻结预测差异。运行时可以通过 `--models-dir` 和 `--cache-dir` 指定模型与缓存目录。已有 CUDA 环境可以沿用 [依赖范围](requirements.txt)，并传 `--device cuda`。

`predict` 可接收多张图片，每张输出一行 JSON。正常输出包含 `model`、解析后的 `model_version`、`label` 和 `decision_scores`。分数用来排序类别，健康分在健康判定阶段用作相对零点。遇到坏图会输出该图的 `error`，并继续处理后续图片。若有坏图，整次命令最终返回非零退出码。手机照片默认按 EXIF 转正，`--orientation raw` 使用冻结回测的原始方向。

## 重跑 PlantCity 历史回测

从 [PlantCity v4 数据页](https://data.mendeley.com/datasets/w8kh2xkspx/4) 下载发布方的 `test.zip`。解压后，将包含 `test/` 子目录的解压目录传给 `--images-root`。保留压缩包内原来的文件名和路径。

```powershell
& $python -m robustleaflab evaluate results/plantcity-v4-replay.csv --images-root .\PlantCity-v4 --model current --device cpu --check-frozen
```

输出 JSON 包含逐图预测、准确率、混淆矩阵和冻结一致性检查。整套 PlantCity 清单为 **872 张**，期望 AC42、v0.1.0 和 v0.2.0 分别判对 **513、640、644** 张。对同一清单切换 `--model ac42` 或 `--model v0.1.0` 可复跑两个对照。完整的 2,285 张历史核对还包括其他来源，其中本仓库提供端到端清单的是 PlantCity。

自定义 CSV 支持 `image,sha256,source,true_label` 四列，`image` 为相对路径。传 `--check-frozen` 时还需要所选模型的 `expected_<model>` 列，例如 `expected_current`。不传 `--images-root` 时使用清单所在目录。评估默认保留历史原始方向，评估新手机照片时可显式用 `--orientation exif`。缺图、坏图或哈希不符时，评估会停止，分母保持与清单一致。

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

默认模型复用 v0.1.0 的基础网络与三个头，再用 seed 42、45、46 的复核头投票。第一轮投票可将原预测健康的图片复核为早疫。第二轮仅复核原已判病害的图片，用不同输出处理早疫改晚疫和晚疫改早疫。至少两票才改判。阈值由冻结开发集按预设召回和种子规则确定。锁定候选与 v0.1.0 的历史回测比较为发布提供依据。 [v0.2.0 发布清单](releases/v0.2.0.json) · [复核权重及阈值](robustleaflab/directional_weights/manifest.json)

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

## 测试与第三方来源

在已安装依赖的环境运行 `python -m unittest discover -s tests -v`，可检查读图、评估和投票的边界行为。样例照片和 DINOv2 的来源及许可证见 [第三方材料说明](THIRD_PARTY_NOTICES.md)。
