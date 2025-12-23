# 🛠️ Annotation Converters (标注格式转换工具)

本目录提供了将 HaGRID 数据集的原生 **JSON** 标注格式转换为深度学习领域最常用的 **YOLO** 和 **COCO** 格式的工具脚本。
!请注意! `convert_utils.py` 与 `dataset.py` 实现作用多有相似之处，具体区别请见dataset文件夹下``

## 📂 文件列表与功能

| 文件名 | 功能描述 | 适用框架 |
| :--- | :--- | :--- |
| **`hagrid_to_yolo.py`** | 将标注转换为 YOLO 格式 (`.txt`) | YOLOv5, YOLOv8, YOLOv10, Ultralytics |
| **`hagrid_to_coco.py`** | 将标注转换为 COCO 格式 (`.json`) | Detectron2, MMDetection, TensorFlow Object Detection API |
| **`convert_utils.py`** | **核心工具库**。负责读取配置、解析原始 JSON、并过滤掉磁盘上不存在的图片。 | (被上述两个脚本调用) |

---

## 📐 坐标系转换说明

理解不同格式的坐标定义对于训练至关重要。

### 1. HaGRID 原生 (Source)
*   **格式**: `[x_min, y_min, width, height]` (归一化或绝对坐标，取决于具体版本)
*   **原点**: 图像左上角。

### 2. YOLO 格式 (Target)
*   **格式**: `[class_id, x_center, y_center, width, height]`
*   **特点**:
    *   **归一化**: 所有数值在 0-1 之间。
    *   **中心点**: 使用 BBox 的中心，而非左上角。
    *   **独立文件**: 每张图片对应一个同名的 `.txt` 文件。

### 3. COCO 格式 (Target)
*   **格式**: `[x_min, y_min, width, height]` (绝对像素坐标)
*   **特点**:
    *   **单文件**: 整个数据集的标注汇总在一个巨大的 `.json` 文件中。
    *   **绝对坐标**: 数值是像素值（如 1024, 768），非归一化。

---

## 🚀 使用指南

在使用这些脚本之前，请确保你已经通过根目录的 `download.py` 下载了数据集，并且配置文件（如 `config/*.yaml`）中的路径已正确指向下载的数据。

### 1. 转换为 YOLO 格式 (`hagrid_to_yolo.py`)

此脚本会遍历指定的数据集目录，为每一张存在的图片生成对应的 `.txt` 标签文件。

```bash
# 示例命令 (具体参数请参考脚本内的 argparse 定义)
python converters/hagrid_to_yolo.py --path_to_config config/your_config.yaml
```

**生成的目录结构示例**:
```text
dataset/
├── train/
│   ├── images/            <-- 原始图片目录 (按类别分文件夹)
│   │   ├── call/
│   │   │   └── 0a1b2c.jpg
│   │   └── like/
│   │       └── ...
│   └── labels/            <-- 脚本自动生成 (与 images 结构对应)
│       ├── call/
│       │   └── 0a1b2c.txt <-- 内容示例: 4 0.51 0.32 0.15 0.20 (类别ID x y w h)
│       └── like/
│           └── ...
```

### 2. 转换为 COCO 格式 (`hagrid_to_coco.py`)

此脚本会生成标准的 COCO `instances_train.json` 和 `instances_val.json`，适用于 Detectron2 或 MMDetection 等框架。

```bash
# 示例命令
python converters/hagrid_to_coco.py --path_to_config config/your_config.yaml
```

---

## 💡 核心逻辑 (`convert_utils.py`)

`convert_utils.py` 是转换过程的“守门员”。它包含一个关键函数 `get_dataframe`，执行以下核心逻辑：

1.  **读取配置**: 获取 `dataset_annotations` (JSON路径) 和 `dataset_folder` (图片路径)。
2.  **解析 JSON**: 将嵌套的 HaGRID JSON 展平为 Pandas DataFrame。
3.  **完整性检查 (关键)**:
    *   它会扫描你的硬盘，获取实际存在的图片列表。
    *   **自动过滤**: 如果某张图片的 JSON 标注存在，但你没有下载这张图（或者文件损坏/丢失），该脚本会自动在标注列表中剔除这条记录。
    *   这有效防止了在后续训练时出现 `FileNotFoundError`。

---

## ⚠️ 注意事项

1.  **类别顺序**: 转换脚本依赖 `constants.py` 中的 `targets` 列表来生成类别 ID (0, 1, 2...)。**请勿随意修改该列表的顺序**，否则会导致生成的标注文件 ID 与模型预测不匹配。
2.  **空数据处理**: 如果图片中没有手（例如类别为 `no_gesture`），YOLO 格式通常会生成一个空的 `.txt` 文件或不生成文件（取决于具体脚本实现），请检查脚本行为是否符合你的训练框架要求。
