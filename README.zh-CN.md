<div align="center">

# COTTON-DETR

### DINOv3 驱动的标注高效田间棉铃状态检测：从冷启动选样到语义解释

**DINOv3-Driven Label-Efficient In-Field Cotton Boll State Detection: From Cold-Start Selection to Semantic Interpretation**

同一自监督模型家族，贯穿图像选择、伪标签验证、目标检测与单铃语义解释。

[English](README.md) · [中文](README.zh-CN.md) · [在线演示](https://huggingface.co/spaces/DEFF819/COTTON-DETR) · [复现说明](REPRODUCIBILITY.md)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)

</div>

![DINOv3 驱动的棉铃监测流程](docs/assets/cotton_detr_pipeline.png)

*论文图 1：数据选样、独立验证的师生学习，以及检测后的 ROI 语义解释。*

## 项目概览

COTTON-DETR 面向标注有限、类别长尾与采集条件变化的田间棉铃监测，将 **数据构建、训练可靠性、目标检测和输出解释** 连接起来。四个阶段复用 DINOv3 模型家族，根据任务使用不同规模的模型与预测头，而非由同一个网络或检查点承担所有功能。

两轮选样从 **3,982 张训练候选图像中标注 608 张**，其余 **3,374 张** 用于半监督训练。最明显的性能收益是 **跨域泛化**：相对同架构纯监督基线，合并外部域 mAP 从 **0.462 提升至 0.548**。独立验证器同时缓解自训练中的稀有类退化；检测后，冻结的 ROI 特征进一步支持开裂与损伤属性预测。

[Hugging Face 演示](https://huggingface.co/spaces/DEFF819/COTTON-DETR)提供 **20 张预计算示例**，支持交互检测框、单铃语义、中英文切换和结果导出。当前**不支持上传新图片实时推理**。精选示例用于定性展示，不代表完整测试集表现。

## DINOv3 的四个角色

| 阶段 | 表征 | 作用 |
|---|---|---|
| **选样** | 冻结 ViT-B/16 全局特征 | 不依赖任务检测器选择 500 张覆盖图像，再根据首轮人工标签定向补充 108 张图像。 |
| **验证** | 冻结 ViT-S/16 ROI 特征 | 在半监督训练中独立判断教师候选框类别并筛选伪标签。 |
| **检测** | DINOv3 蒸馏 ViT-Tiny | DEIMv2-DINOv3-S 输出位置与四类检测标签；检测器为 9.69M 参数。 |
| **解释** | 冻结 ViT-S/16 ROI 特征 | 类别平衡 MLP 预测开裂属性；训练特征原型预测损伤类型与程度。 |

检测类别为 **残次铃（Defected）、棉花花朵（Flower）、全开铃（Fully opened）和半开铃（Partially opened）**。Flower 指花器官；**闭合（Closed）属于语义开裂状态，不是第五个检测类别**。

### 训练阶段的独立验证

![检测器架构及冻结的伪标签验证分支](docs/assets/cotton_detr_architecture.png)

*论文图 7：上部为检测器，下部为训练时使用的冻结验证分支。*

EMA 教师从弱增强无标注图像中提出候选框；冻结 DINOv3 ROI 特征与经过校准的分类头独立确定类别和接受条件。学生使用人工标注及强增强图像上的有效伪标签进行训练。

**验证器后验决定接受的类别；教师置信度和类别权重决定其损失贡献。** 伪标签验证器是训练组件。推理时，由训练后的检测器输出框与类别，再进行 ROI 语义预测。

## 主要实验依据

除单独说明外，以下检测结果均为 **3 个随机种子的均值 ± 标准差**。COCO mAP 指 AP@[0.50:0.05:0.95]。这些数值来自论文实验，不是在线示例的统计结果。

### 跨域泛化

![外部域性能保持与不同标注预算下的增益](docs/assets/cotton_detr_crossdomain.png)

*论文图 12：星号表示单次运行参考模型；同架构对照使用 3 个种子。*

| 模型，608 张标注图像 | OUT100 mAP | GH80 mAP | 合并 mAP | 性能保持率 |
|---|---:|---:|---:|---:|
| 同架构纯监督基线 | 0.477 ± 0.026 | 0.397 ± 0.046 | 0.462 ± 0.034 | 0.780 ± 0.061 |
| **COTTON-DETR** | **0.570 ± 0.008** | **0.472 ± 0.027** | **0.548 ± 0.005** | **0.908 ± 0.016** |

*来源：表 4。保持率 = 外部域 mAP / 域内 mAP。*

外部评测共 **180 张图像、510 个框**：OUT100 为 100 张露田图像，GH80 为 80 张温室图像且不含花朵。**跨域增益 +0.086** 约为同预算下 **域内增益 +0.012** 的 7 倍。这里验证的是棉铃任务在不同采集条件下的迁移，尚未验证跨作物泛化。

### 稀有类监督与保护

| 608 张标注预算下的对照 | 残次铃 F1 | 解释 |
|---|---:|---|
| 自训练不使用 → 使用验证器 | 0.561 ± 0.043 → **0.606 ± 0.013** | 缓解稀有类退化，并降低训练种子间波动。 |
| 随机选样 → DINOv3 冷启动选样 | 0.591 ± 0.055 → **0.618 ± 0.005** | 纯监督选样消融：3 个随机子集，对比固定 DINOv3 子集上的 3 个训练种子。 |

*来源：表 3。两行属于不同消融实验，不能串联为同一次实验的累计增益。*

验证器相对无验证自训练改善了 F1，但仍**未超过该预算下纯监督的 0.618**。选样对照两组的波动来源不同，也不能直接解读为配对的重复子集方差检验。

### 标注预算与腐蚀鲁棒性

![三种子标注效率和鲁棒性对照](docs/assets/cotton_detr_label_efficiency.png)

*论文图 8：域内增益集中于最低预算，三个预算的腐蚀鲁棒性均有改善。*

| 标注数 | 纯监督 mAP | COTTON-DETR mAP | 论文报告 ΔmAP | 纯监督 rPC | COTTON-DETR rPC |
|---:|---:|---:|---:|---:|---:|
| 152 | 0.534 ± 0.008 | 0.578 ± 0.005 | +0.043 | 0.790 ± 0.004 | 0.824 ± 0.008 |
| 304 | 0.575 ± 0.004 | 0.585 ± 0.007 | +0.010 | 0.792 ± 0.005 | 0.846 ± 0.004 |
| 608 | 0.592 ± 0.009 | 0.604 ± 0.005 | +0.012 | 0.835 ± 0.015 | 0.856 ± 0.005 |

*来源：表 2。差值根据未舍入数据计算。架构、标注子集和增强方式一致；COTTON-DETR 额外使用 3,374 张无标注图像。测试集共 523 张图像。*

`mPC` 为 15 种腐蚀类型 × 5 个严重程度的平均 mAP；`rPC = mPC / 干净测试集 mAP`。mAP 提升不代表所有类别的 F1 均提升。论文中的配对检验仅使用 3 个种子，未做多重比较校正。

<details>
<summary>跨架构腐蚀鲁棒性对照</summary>

![代表性单次运行的腐蚀鲁棒性对照](docs/assets/cotton_detr_robustness.png)

*图 10 使用代表性单次运行，不是表 2 的三种子均值。* 基于 DINOv3 的检测器在合成腐蚀下保持较高性能；同架构实验进一步区分半监督学习带来的较小额外收益。合成腐蚀是受控退化基准，不能替代真实域偏移评测。

</details>

## 单铃 ROI 语义解释

语义组件将检测 ROI 扩展为 **开裂状态、开裂比例、损伤类型、损伤程度与结构化说明**：

1. 冻结 DINOv3 ViT-S/16 编码裁剪图像。
2. 类别平衡 MLP 预测开裂状态和比例。比例按预测状态约束为：花朵 0%，闭合 0–15%，部分开裂 15–65%，完全开裂 65–100%。
3. 损伤有无遵循检测类别；残次铃的类型与程度通过训练集正类特征原型匹配获得。
4. 文字由属性和模板确定性组合。**Gemini 3 Flash 用于离线属性监督；推理不调用 VLM，也不自由生成文本。**

仅使用 **608 张训练图像的 1,203 个标注 ROI** 训练语义头及构建原型。开裂比例估计单个 ROI 的可见棉絮外露程度，不是整幅图像或田块的开铃率。

### 属性一致性与评估范围

表 5 以 **523 张测试图像中的全部 1,026 个真实框 ROI** 为对象，与 Gemini 属性标签比较。类别相关决策使用真实位置与类别，因此评估的是已知目标条件下的属性预测，而非端到端检测。

| 组件 | 开裂状态 macro-F1 ↑ | 开裂比例 MAE ↓ | 损伤程度 macro-F1，4 类 ↑ |
|---|---:|---:|---:|
| 确定性类别映射 | 0.715 | 8.80 | 0.449 |
| 原始多头 MLP | 0.794 | 4.96 | 0.448 |
| 固定开裂预测 + 损伤多数类 | 0.794 | 4.65 | 0.449 |
| **完整语义组件** | **0.794** | **4.65** | **0.610** |

MAE 单位为 **百分点**。新增多数类对照保持开裂预测、比例约束和损伤有无决策不变；与完整组件比较后可单独考察损伤原型的贡献：严重程度 macro-F1 从 0.449 提升到 0.610，未舍入增益约为 0.162。

![开裂与损伤属性相对 Gemini 标签的一致性](docs/assets/cotton_detr_semantic.png)

*论文图 13：行为参考标签，列为预测；单元格显示样本数及行内比例。*

- **全体测试 ROI：** 表 5 的严重程度 macro-F1 为 **0.610，采用四类**，包含 935 个无损伤目标。
- **仅残次铃 ROI：** 图 13 的损伤面板使用 91 个残次铃测试 ROI。轻度、中度、重度 **三类 macro-F1 为 0.481**，与表 5 不同。
- **仅正确检出的目标：** 置信度阈值 0.65 下，891 个类别正确且 IoU ≥ 0.5 的匹配目标，其开裂/严重程度 macro-F1 为 **0.848 / 0.699**。该指标排除了 450 个假阳性和 135 个漏检，不是端到端语义分数。

这些指标衡量与 VLM 标签的一致性。独立专家标注和农学准确性校准仍属于后续工作。损伤有无受类别约束，原型相似度不是概率，也不是专家诊断。

## 代码与复现

```text
configs/cotton_detr/     半监督及语义配置
engine/                 检测器与半监督训练实现
selection/              冷启动选样算法
semantic/               ROI 语义编码库
tools/selection/        特征提取和选样命令
tools/verifier/         验证器训练及校准
tools/semantic/         语义训练、评估与导出
tools/evaluation/       腐蚀评估工具
prompts/                ROI 属性标注提示词
scripts/                复现预检查
third_party/licenses/   上游许可
```

训练使用 Linux 和 NVIDIA GPU。先安装与 CUDA 环境兼容的 PyTorch/torchvision，再安装复现依赖：

```bash
git clone https://github.com/hwzhanng/COTTON-DETR.git
cd COTTON-DETR
python -m venv .venv
source .venv/bin/activate
# 先安装与 CUDA 环境兼容的 torch/torchvision。
pip install -r requirements-repro.txt
python scripts/check_repro_package.py \
  --config configs/cotton_detr/cotton_detr_dinov3_tiny_ssod.yml
```

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) 提供选样、验证器校准、训练、EMA 教师评估、数据组织与语义组件命令。论文的**全局冷启动选样使用 ViT-B/16**；**验证器及语义 ROI 使用 ViT-S/16**，各阶段需配置对应骨干。

当前 Git 仓库包含代码、配置模板、提示词和精选论文图件，不在该仓库内打包数据或检查点；参见[数据说明](data/README.md)和[权重配置](weights/pretrained/README.md)。DINOv3 原始权重需遵守上游访问与许可条件获取。仓库不包含完整论文 PDF、原始 VLM 响应或本地实验归档。

## 论文与引用

**DINOv3-Driven Label-Efficient In-Field Cotton Boll State Detection: From Cold-Start Selection to Semantic Interpretation**

论文公开链接及完整引用信息将在可用时补充。仓库使用的论文图件来源见 [docs/assets](docs/assets/README.md)。

## 许可与致谢

COTTON-DETR 原创增量和发布材料沿用仓库现有的 [CC BY-NC 4.0 许可](LICENSE)。非商业使用需要署名；商业使用需要作者事先书面许可。

DEIM/DEIMv2、D-FINE、RT-DETR 和 DINOv3 保留各自上游许可，详见 [NOTICE.md](NOTICE.md)、[OPEN_SOURCE_AUDIT.md](OPEN_SOURCE_AUDIT.md) 和 [third_party/licenses](third_party/licenses/)。感谢这些项目提供的方法与实现。
