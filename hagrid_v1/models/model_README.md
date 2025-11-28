我强烈怀疑这个介绍不一定正确
# 🧠 Models Architecture (模型架构定义)

本目录负责定义项目中所有神经网络的**结构与前向传播逻辑**。
它不包含训练循环或数据处理代码，仅专注于回答“网络长什么样”这个问题。

## 📂 目录结构与层次

本项目采用了**模块化**的设计，将分类任务与检测任务的实现物理隔离，并通过统一的基类和注册表进行管理。

```text
models/
├── __init__.py                # 1. 注册表：汇总所有模型，对外暴露接口
├── model.py                   # 2. 基类：定义所有模型的抽象父类 (HaGRIDModel)
├── classifiers/               # --- 图像分类模型 (Image Classification) ---
│   ├── base_model.py          # 3. 通用适配器：封装 ResNet, MobileNet 等标准 CNN
│   └── vit_model.py           # 4. Transformer：专门封装 Vision Transformer (ViT)
└── detectors/                 # --- 目标检测模型 (Object Detection) ---
    └── ssd_mobilenetv3.py     # 5. SSD 检测器：SSDLite + MobileNetV3 的完整实现
```

---

## 🏗️ 核心组件解析

### 1. 抽象基类 (`models/model.py`)
*   **类名**: `HaGRIDModel`
*   **角色**: 家族的“老祖宗”。
*   **作用**:
    *   定义了所有模型必须遵守的**统一接口**。
    *   无论子类是分类模型还是检测模型，都必须继承此将被 `Trainer` (训练器) 统一调用。
    *   包含了通用的初始化逻辑，如权重加载接口。

### 2. 注册表 (`models/__init__.py`)
*   **角色**: 整个文件夹的**调度中心**。
*   **作用**:
    *   **汇集**: 从子文件夹 (`classifiers`, `detectors`) 导入具体的模型类。
    *   **映射**: 维护 `classifiers_list` 和 `detectors_list` 两个字典，建立 **字符串名称** (如 `"ResNet18"`) 到 **Python 类** 的映射关系。
    *   **工作流**: 当 `config.yaml` 中设置 `model: name: "ResNet18"` 时，`utils.py` 中的构建函数会查阅此处的字典来实例化模型。

---

## 🧩 子模块详解

### A. 分类器 (`models/classifiers/`)
> 🎯 **适用场景**: `demo_ff.py` (全帧分类)，输入整张图片，输出一个手势类别。

| 文件名 | 功能描述 |
| :--- | :--- |
| **`base_model.py`** | **万能适配器 (Generic CNN Wrapper)**。<br>它不从零实现网络，而是调用 `torchvision` 库中成熟的 CNN 模型 (如 ResNet, MobileNet, ConvNeXt)。<br>**核心逻辑**: 加载预训练骨干网 -> 替换最后一层全连接层 (FC) -> 输出 **34 类** (HaGRID 手势数量)。 |
| **`vit_model.py`** | **ViT 专用封装 (Vision Transformer)**。<br>由于 Transformer 架构 (Patch Embedding, Attention) 与传统 CNN 差异较大，输入输出处理逻辑不同，因此单独使用此文件进行封装。 |

### B. 检测器 (`models/detectors/`)
> 🎯 **适用场景**: `demo.py` (手势检测)，输入图片，输出手势的边界框 (BBox) 和类别。

| 文件名 | 功能描述 |
| :--- | :--- |
| **`ssd_mobilenetv3.py`** | **重型武器 (SSDLite Implementation)**。<br>这是本项目中代码最复杂的文件，完整实现了 **SSDLite** 检测算法，包含四个核心部分：<br>1. **Backbone**: 使用 MobileNetV3 提取特征。<br>2. **Extra Layers**: 额外的下采样卷积层，用于多尺度特征提取。<br>3. **Heads**: 回归头 (预测坐标偏移) + 分类头 (预测类别概率)。<br>4. **Anchor Generator**: 锚框生成逻辑。 |

---

## 🚀 扩展指南 (How to Add New Models)

如果你想尝试一个新的网络架构（例如 YOLOv12 或 EfficientNet），请按以下步骤操作：

1.  **定义网络**:
    *   如果是 **分类模型**：在 `models/classifiers/` 下新建文件（或直接修改 `base_model.py` 添加支持）。
    *   如果是 **检测模型**：在 `models/detectors/` 下新建文件。
    *   **注意**: 新的类必须继承自 `models.model.HaGRIDModel`。

2.  **注册模型**:
    *   打开 `models/__init__.py`。
    *   导入你新写的类。
    *   将其加入到 `classifiers_list` (分类) 或 `detectors_list` (检测) 字典中。

3.  **配置参数**:
    *   在 `config/` 目录下复制一份现有的 `.yaml` 文件。
    *   将 `model.name` 修改为你刚才在字典中注册的 **键名 (Key)**。