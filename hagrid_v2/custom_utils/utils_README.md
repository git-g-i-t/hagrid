# 🛠️ Custom Utilities (核心工具箱)

本目录存放了项目训练、推理和分布式计算的**核心后端逻辑**。
这里的代码不定义神经网络结构（在 `models/` 中），也不处理原始数据（在 `dataset/` 中），而是负责**将数据和模型连接起来，驱动整个训练过程**。
其实就是定义了很多函数。
## 📂 文件概览

| 文件名 | 角色 | 核心功能 |
| :--- | :--- | :--- |
| **`train_utils.py`** | 🚂 **训练引擎** | 封装了训练循环 (Epoch/Batch)、验证逻辑、优化器加载及 Checkpoint 保存。 |
| **`utils.py`** | 🔧 **通用工具** | 提供模型构建工厂 (`build_model`)、评价指标计算 (F1/Accuracy) 及日志配置。 |
| **`ddp_utils.py`** | ⚡ **分布式助手** | 处理多 GPU 并行训练 (DDP) 的初始化、通信与资源释放。 |

---

## 📖 详细模块解析

### 1. `train_utils.py` (The Training Engine)
这是本仓库中最复杂、最重要的文件。它定义了 `Trainer` 类，这是训练任务的总指挥。

*   **`class Trainer`**:
    *   **`train()`**: 主循环，管理 Epoch 迭代。
    *   **`train_one_epoch()`**: 核心逻辑。执行 `Forward` -> `Loss Calculation` -> `Backward` -> `Optimizer Step`。
    *   **`test()` / `validate()`**: 在验证集/测试集上评估模型性能，计算 Loss 和 Metrics。
    *   **`save_model()`**: 将训练好的权重保存为 `.pth` 文件。
*   **辅助函数**:
    *   `load_train_objects()`: 统一加载 DataLoader 和 Model 实例。
    *   `load_train_optimizer()`: 根据配置初始化优化器 (SGD/Adam) 和 学习率调度器 (Scheduler)。

### 2. `utils.py` (General Utilities)
这是一个“百宝箱”，存放被多个模块调用的通用函数。

*   **`build_model(conf)` (关键)**:
    *   **工厂模式**。它读取配置文件中的字符串 (如 `"ResNet18"`)，并在 `models/` 目录中找到对应的类进行实例化。
    *   它是连接 `config/*.yaml` 和 `models/*.py` 的桥梁。
*   **Metrics (指标)**:
    *   `F1ScoreWithLogging`: 自定义的 F1 分数计算器，支持二分类和多分类任务，并集成了日志打印功能。
*   **其他**:
    *   日志配置 (Logging setup)。
    *   随机种子设置 (Seed setting)，确保实验可复现。

### 3. `ddp_utils.py` (Distributed Data Parallel)
仅在使用 `run.sh` 进行多显卡训练时被调用。

*   **`ddp_setup()`**: 初始化进程组，设置 `MASTER_ADDR` 和 `MASTER_PORT`，分配 GPU 设备。
*   **`destroy_process_group()`**: 训练结束后清理进程，释放显存资源。
*   **`reduce_loss()`**: 在计算 Loss 时，将所有显卡上的计算结果汇总取平均值，确保梯度更新同步。

---

## 🔄 它们如何协作？ (Work Flow)

当你在根目录运行 `python run.py` 时，数据流向如下：

1.  **启动**: `run.py` 调用 **`utils.py`** 中的 `build_model` 创建模型结构。
2.  **准备**: `run.py` 调用 **`ddp_utils.py`** (如果是多卡) 初始化环境。
3.  **组装**: `run.py` 调用 **`train_utils.py`** 中的 `load_train_objects` 准备数据和优化器。
4.  **循环**: `run.py` 初始化 **`Trainer`** 类 (`train_utils.py`)，开始死循环：
    *   Trainer 从 `dataset.py` 拿数据。
    *   Trainer 喂给 Model。
    *   Trainer 调用 `utils.py` 算分。
    *   Trainer 更新参数。
5.  **结束**: 保存模型，清理 DDP 进程。

---

## 💡 开发者建议

*   **如果你想修改 Loss 计算逻辑或训练流程**：请查看 `train_utils.py` 里的 `train_one_epoch` 方法。
*   **如果你添加了一个新的模型架构**：需要在 `utils.py` 的 `build_model` 函数中注册你的新模型，否则配置文件无法识别。
*   **如果你想修改评估指标 (如增加 Recall)**：请修改 `utils.py`。