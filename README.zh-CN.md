<div align="center">

# COTTON-DETR

### 标注高效的田间棉铃多状态检测

**由同一 DINOv3 谱系驱动的检测器无关冷启动选样与可靠性受控半监督学习**

[English](README.md) | [中文](README.zh-CN.md) | [复现说明](REPRODUCIBILITY.md) | [开源许可](LICENSE)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)

</div>

<p align="center">
  <img src="docs/assets/cotton_detr_pipeline.png" alt="COTTON-DETR 端到端管线" width="100%">
</p>

## 项目概览

田间图像精细标注成本高，而农业目标通常密集、细粒度且具有明显长尾分布。COTTON-DETR 是一套标注高效的检测管线，将同一 DINOv3 自监督预训练谱系复用于四个角色：检测器无关的冷启动选样、伪标签验证、蒸馏 DETR 骨干，以及检测后的 ROI 语义编码。

在 3,982 张田间图像中仅标注 608 张（15%）时，COTTON-DETR 在域内测试集取得 **0.604 mAP** 和 **0.856 rPC**。半监督的主要价值体现在分布偏移下：相对同架构纯监督模型，无标注数据使跨域 mAP 提升 **+0.086**，同时保留 **91%** 的域内精度。本仓库仅发布代码、脱敏配置、提示词与复现工具，不包含私有数据、权重、检查点或实验归档。

## 核心特点

- **检测器无关冷启动。** 在任何任务检测器训练之前，冻结 DINOv3 ViT-B/16 特征先选择覆盖整体分布的标注头部，再向稀缺残次类扩展。
- **可靠性受控半监督检测。** 独立冻结的 DINOv3 验证器对教师框重新判类并筛除不可靠候选；类别由验证器决定，损失权重仍由教师证据决定。
- **有边界的标签效率结论。** 三种子实验显示，域内 mAP 增益在 152 张标注时最大（+0.043），预算增大后收窄；腐蚀鲁棒性在三个预算下均保持正增益。
- **跨域泛化。** 在 608 张标注设置下，COTTON-DETR 相对同架构纯监督基线将合并外部域 mAP 从 0.462 提升到 0.548。
- **超越检测框的语义输出。** 冻结 DINOv3 ROI 编码器输出开裂状态、开裂比例、损伤亚型和严重程度等结构化农艺线索。

## 方法

<p align="center">
  <img src="docs/assets/cotton_detr_architecture.png" alt="COTTON-DETR 可靠性受控半监督架构" width="100%">
</p>

COTTON-DETR 采用 teacher-student DETR。EMA 教师从弱增强无标注图像中产生候选框；冻结 DINOv3 ViT-S/16 验证器独立校验 ROI 类别并拒绝不可靠候选；学生使用强增强无标注图像上的已验证伪标签，以及标注图像上的真值框联合优化。

该设计将检测器自训练通常耦合的两个判断分开：

1. **这个 ROI 属于哪一类？** 由 DINOv3 验证器给出类别判断。
2. **这个样本应贡献多大损失？** 由教师分数给出权重。

这种解耦针对长尾类别中的确认偏置，同时避免把验证器扩展为第二个可训练检测器。

## 实验结果

### 标签效率与腐蚀鲁棒性

<p align="center">
  <img src="docs/assets/cotton_detr_label_efficiency.png" alt="三种子标签效率与鲁棒性结果" width="100%">
</p>

受控比较使用相同的 DEIMv2-DINOv3-S 架构、标注子集、数据增强与评测划分；半监督行额外使用 3,374 张无标注图像。下表均为 523 张测试图像上的三种子均值 ± 标准差。

`mPC` 表示各腐蚀类型与强度下的平均 mAP，`rPC = mPC / clean mAP`。

| 方法 | 标注图像数 | mAP ↑ | mPC / rPC ↑ |
|---|---:|---:|---:|
| 纯监督 | 152 | 0.534 ± 0.008 | 0.422 ± 0.006 / 0.790 ± 0.004 |
| **COTTON-DETR** | **152** | **0.578 ± 0.005** | **0.476 ± 0.008 / 0.824 ± 0.008** |
| 纯监督 | 304 | 0.575 ± 0.004 | 0.456 ± 0.005 / 0.792 ± 0.005 |
| **COTTON-DETR** | **304** | **0.585 ± 0.007** | **0.495 ± 0.005 / 0.846 ± 0.004** |
| 纯监督 | 608 | 0.592 ± 0.009 | 0.494 ± 0.004 / 0.835 ± 0.015 |
| **COTTON-DETR** | **608** | **0.604 ± 0.005** | **0.517 ± 0.003 / 0.856 ± 0.005** |

按各方法报告的 best-test checkpoint 进行相同标注预算比较，COTTON-DETR 于 152/608 张标注时取得 0.578/0.604 mAP；Soft Teacher 为 0.496/0.541，Semi-DETR 为 0.497/0.547。这一 checkpoint 级比较说明，通用 SSOD 配方不能直接解决该田间长尾场景。

### 组件证据

| 组件 | 608 张标注下的三种子证据 | 结论边界 |
|---|---|---|
| DINOv3 验证器 | 残次铃 F1：0.561 ± 0.043 → **0.606 ± 0.013** | 主要修复并稳定稀缺类；mAP 仅由 0.599 变为 0.604。 |
| DINOv3 冷启动选样 | 残次铃 F1：0.591 ± 0.055 → **0.618 ± 0.005** | 主要改善稀缺类覆盖和方差，而非大幅抬高平均 mAP。 |
| 无标注训练数据 | 152/304/608 张标注下 rPC 分别提升 +0.035、+0.053、+0.021 | 在同一架构内带来较小但一致的鲁棒性增益。 |

### 未见域泛化

<p align="center">
  <img src="docs/assets/cotton_detr_crossdomain.png" alt="跨域精度保持率与半监督增益" width="100%">
</p>

在两个未见田间域上，COTTON-DETR 取得 **0.548 ± 0.005 mAP** 和 **0.908 ± 0.016 rExt**；同架构纯监督基线为 **0.462 ± 0.034 mAP** 和 **0.780 ± 0.061 rExt**，其中 `rExt = 外部域 mAP / 域内 mAP`。608 张标注时，+0.086 的跨域 mAP 增益约为 +0.012 域内增益的七倍。外部评测规模为 180 张图像、510 个框，更广泛的跨地点与跨作物验证仍是后续工作。

### 腐蚀鲁棒性

<p align="center">
  <img src="docs/assets/cotton_detr_robustness.png" alt="合成腐蚀下的准确率与鲁棒性" width="100%">
</p>

跨架构鲁棒性趋势主要与 DINOv3 蒸馏检测骨干相关；在相同架构内部，半监督学习再贡献较小的正增量。ImageNet-C 风格合成腐蚀在本文中是成像退化的可控代理，不能替代真实域偏移评测。

### ROI 语义编码

<p align="center">
  <img src="docs/assets/cotton_detr_semantic.png" alt="COTTON-DETR 语义编码示例" width="100%">
</p>

可选语义组件把检测 ROI 转换为开裂状态、开裂比例、损伤亚型和严重程度。以 ROI 级 VLM 标签为参照时，原型匹配将损伤严重程度一致性 F1 从 0.448 提升至 **0.611**。这些输出是结构化视觉线索，不是专家病理诊断，也不等同于人工真值准确率。

## 仓库结构

```text
COTTON-DETR/
├── configs/cotton_detr/       # 公开 SSOD 与语义配置模板
├── engine/                    # 检测器、损失、求解器与 SSOD 运行时
├── selection/                 # 冷启动选样算法
├── semantic/                  # ROI 语义编码库
├── tools/selection/           # 特征提取与选样命令行工具
├── tools/verifier/            # 验证器训练与阈值校准
├── tools/evaluation/          # 腐蚀指标汇总工具
├── tools/semantic/            # 语义训练、校验与导出
├── prompts/                   # 开源 ROI 语义标注提示词
├── scripts/                   # 复现预检脚本
├── REPRODUCIBILITY.md         # 端到端复现实验协议
└── third_party/licenses/      # 上游许可证与声明
```

## 安装

训练环境为 Linux + NVIDIA GPU。请先安装与 CUDA 驱动匹配的 PyTorch，再安装其余依赖。

```bash
git clone https://github.com/hwzhanng/COTTON-DETR.git
cd COTTON-DETR

python -m venv .venv
source .venv/bin/activate

# 仅为示例，请根据 CUDA 环境选择对应 wheel。
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-repro.txt
```

## 数据与权重

默认配置使用 COCO 格式标注：

```text
data/
├── candidate/images/
├── labeled/images/
├── unlabeled/images/
├── val/images/
└── annotations/
    ├── candidate_train.json
    ├── coverage_head_labeled.json
    ├── labeled_train.json
    ├── unlabeled_train.json
    ├── val.json
    └── test.json
```

使用者需要按上游许可自行获取 DEIMv2-DINOv3 检测器初始化权重与 DINOv3 模型文件，并用自己的标注数据训练验证器头。训练前运行预检：

```bash
python scripts/check_repro_package.py \
  --config configs/cotton_detr/cotton_detr_dinov3_tiny_ssod.yml
```

仅在准备目录结构时使用 `--allow-missing`。

## 复现流程

### 1. 冷启动选样

```bash
python tools/selection/extract_dinov3_features.py \
  --image-dir data/candidate/images \
  --model-dir weights/pretrained/dinov3_vits16 \
  --output-dir outputs/selection/features

python tools/selection/select_cold_start.py \
  --method cluster_head \
  --features outputs/selection/features/dinov3_features.npy \
  --paths outputs/selection/features/dinov3_paths.npy \
  --candidate-coco data/annotations/candidate_train.json \
  --exclude-coco data/annotations/val.json \
  --exclude-coco data/annotations/test.json \
  --budget 500 \
  --output-dir outputs/selection/coverage_head
```

完成覆盖头部标注后，继续执行 `cluster_tail` 并构建嵌套的 152/304/608 子集。完整命令和防数据泄漏检查见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md#cold-start-selection)。

### 2. 验证器校准

```bash
python tools/verifier/train_linear_verifier.py \
  --train-coco data/annotations/labeled_train.json \
  --train-image-dir data/labeled/images \
  --val-coco data/annotations/val.json \
  --val-image-dir data/val/images \
  --model-dir weights/pretrained/dinov3_vits16 \
  --target-precision 0.85 \
  --output-dir outputs/verifier
```

### 3. 半监督训练

```bash
bash train_guarded.sh \
  -c configs/cotton_detr/cotton_detr_dinov3_tiny_ssod.yml \
  -t weights/pretrained/deimv2_dinov3_s_coco_tuning.pth \
  -d cuda \
  --seed 0 \
  --use-amp
```

### 4. Teacher EMA 评测

```bash
python tools/evaluate_ssod_teacher.py \
  -c configs/cotton_detr/cotton_detr_dinov3_tiny_ssod.yml \
  --checkpoint outputs/cotton_detr_ssod/best_teacher_ema.pth \
  --output-dir outputs/cotton_detr_ssod/eval \
  -d cuda
```

### 5. 可选语义组件

```bash
python tools/semantic/validate_semantic_labels.py \
  data/semantic/semantic_labels_train.jsonl \
  --fail-on-error

python tools/semantic/train_semantic_head.py \
  --config configs/cotton_detr/semantic_sidecar_template.yaml
```

开源提示词与 JSONL 模式见 [tools/semantic](tools/semantic/)。完整种子方案、权重布局、评测口径与结果归档清单见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 开源范围

本仓库包含可执行的方法链路、脱敏配置、提示词模板与论文图。以下内容不公开：

- 私有棉花数据集与标注；
- 预训练模型文件与验证器头；
- 检查点、日志、特征缓存和原始实验结果；
- 私有论文草稿、审稿材料与 VLM 响应。

生成的选样结果、语义标签与评测产物已被 Git 忽略，不应提交到公开仓库。

## 引用

学术使用本仓库必须引用我们的文章。文章链接与完整引用信息将在论文发表后补充。

- 文章：[COTTON-DETR]()

## 许可与第三方声明

COTTON-DETR 原创增量代码与发布材料采用 [CC BY-NC 4.0](LICENSE) 许可。非商业使用须署名；商业使用需事先获得 COTTON-DETR 作者书面许可。

第三方组件继续遵循各自上游许可证。DEIM、D-FINE 与 RT-DETR 相关材料采用 Apache-2.0；DINOv3 代码与模型材料受 Meta DINOv3 License 约束。具体边界见 [NOTICE.md](NOTICE.md)、[OPEN_SOURCE_AUDIT.md](OPEN_SOURCE_AUDIT.md) 与 [third_party/licenses](third_party/licenses/)。

## 致谢

COTTON-DETR 基于 DEIM/DEIMv2、D-FINE、RT-DETR 与 DINOv3 构建。感谢这些项目的作者公开研究代码与模型生态。
