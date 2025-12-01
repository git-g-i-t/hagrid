# Model Configuration Files (配置文件说明)

本目录存放了用于不同任务（目标检测与图像分类）的模型配置文件。每个 `.yaml` 文件定义了模型的架构、训练超参数、数据增强策略等

## 一、 文件分类与功能解析

这些配置文件主要分为两大类，分别对应仓库中的不同演示脚本。

### 1. 目标检测 (Object Detection)
>  **适用脚本**: `demo.py` (画框 + 识别)

| 配置文件 | 说明 |
| :--- | :--- |
| **`SSDLiteMobileNetV3Large.yaml`** | **核心文件**。使用 SSDLite 检测头 + MobileNetV3 Large 骨干网络。这是该项目做手势检测的主力配置，包含 Anchor、NMS 等检测特有的参数。 |

### 2. 图像分类 (Image Classification)
>  **适用脚本**: `demo_ff.py` (全图识别，无框)

这些模型仅用于对整张图片进行分类，区别在于**速度**与**精度**的权衡：

| 配置文件 | 类型 | 特点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **`MobileNetV3.yaml`** | 轻量级 | ⚡ 速度快，标准版 | 移动端、实时性要求高 |
| **`MobileNetV3_small.yaml`** | 极速版 | 🚀 速度极快，参数最少 | 算力极低的嵌入式设备 |
| **`ResNet18.yaml`** | 基础版 | ⚖️ 结构简单，通用性强 | **入门推荐**，适合学习分类配置结构 |
| **`ResNet152.yaml`** | 高精度 | 🐢 精度高，推理慢 | 服务器端，追求极致准确率 |
| **`ConvNeXt_base.yaml`** | 现代 CNN | 🔥 性能优于 ResNet | 追求 SOTA (State of the Art) 性能 |
| **`VitB16.yaml`** | Transformer | 🤖 基于 Attention 机制 | 学术研究，大数据集训练 |

---

## 二、 建议阅读顺序

第一次阅读本项目代码，建议按以下顺序阅读配置文件：

1.  **`SSDLiteMobileNetV3Large.yaml`** (必读)
    *   这是理解 `demo.py` 运行逻辑的关键。
    *   重点关注 `model` 部分的检测头设置以及 `dataset` 部分的参数。
    *   检测模型的配置通常比分类模型复杂（多了 Anchor、NMS 等参数）。读懂了这个，看其他的就觉得很简单了。

2.  **`ResNet18.yaml`** (选读)
    *   标准的分类模型配置。
    *   阅读它可以理解 `demo_ff.py` 的配置逻辑，并对比“分类任务”与“检测任务”在参数设置上的区别。

3.  **其他文件**
    *   仅在需要更换特定模型架构（如使用 Transformer 或更轻量的模型）时参考。

4. **注**
   *  我只详细注释了`SSDLiteMobileNetV3Large.yaml`，其余均是简略注释，大致内容都是一致的，只是选择的参数略有差别
   *  最推荐的阅读顺序是`SSDLiteMobileNetV3Large.yaml`，`ResNet18.yaml`，`ResNet152.yaml`，`MoblieNetV3_small.yaml`,`MoblieNetV3_large.yaml`