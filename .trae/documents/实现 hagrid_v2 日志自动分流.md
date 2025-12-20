# 实现 TensorBoard 日志自动分流 (Train/Test)

为了解决训练日志和测试日志混淆的问题，我将修改 `hagrid_v2` 中的日志记录逻辑，使其根据当前的运行模式将日志写入不同的子目录。

**注意：严格遵守用户指令，仅修改 `hagrid_v2` 目录下的代码，绝不触碰 `hagrid_v1`。**

## 1. 核心修改点
修改 `hagrid_v2/custom_utils/train_utils.py` 文件中的 `Trainer.__init__` 方法。

目前代码可能直接将 `log_dir` 设为 `work_dir/实验名/logs`。
我将修改为：
- 如果是训练模式 (`run.py -c train`)：路径设为 `work_dir/实验名/logs/train`
- 如果是测试模式 (`run.py -c test`)：路径设为 `work_dir/实验名/logs/test`

## 2. 实施步骤
1.  **读取文件**：读取 `hagrid_v2/custom_utils/train_utils.py`，定位 `SummaryWriter` 的初始化位置。
2.  **修改逻辑**：
    - 在初始化 `SummaryWriter` 之前，判断当前是在训练还是测试。
    - 既然 `Trainer` 类本身没有显式的 `mode` 参数，我需要检查它是如何被调用的，或者通过 `self.config` 或方法调用来推断。
    - **更稳妥的方式**：`Trainer` 在 `__init__` 时并不执行训练或测试，而是在调用 `train()` 或 `test()` 方法时才开始。因此，最佳修改位置是在 `train()` 和 `test()` 方法内部，或者在 `__init__` 中创建两个 Writer（一个 train 一个 val/test），但 TensorBoard 通常建议一个 Writer 对应一个文件夹。
    - **修正方案**：
        - 在 `run.py` 中，`Trainer` 初始化时并没有传入 `mode`。
        - 我将修改 `Trainer` 的 `__init__` 方法，增加一个逻辑：默认初始化到 `logs/`。
        - 但这不够完美。
        - **最佳方案**：修改 `run.py`，在初始化 `Trainer` 之前，根据 `args.command` 修改 `config.experiment_name` 或者直接传参告诉 Trainer 日志存哪。
        - **决定**：修改 `hagrid_v2/custom_utils/train_utils.py`。在 `__init__` 中，检查是否可以区分模式。如果不方便，我将在 `run.py` 中把 `args.command` 传递给 `Trainer`，让 `Trainer` 决定路径。

**经查阅 `run.py` 代码：**
`run.py` 已经在调用 `Trainer` 时传入了 `config`。
我们可以修改 `hagrid_v2/run.py`，在传递 `config` 给 `Trainer` 之前，临时修改一下 `config` 中的日志路径，或者给 `Trainer` 加一个 `log_subdir` 参数。

**最终方案**：
1. 修改 `hagrid_v2/custom_utils/train_utils.py`：
   - 更新 `Trainer` 的 `__init__` 方法，接受一个可选的 `log_subdir` 参数。
   - 在创建 `SummaryWriter` 时，将这个子目录拼接到路径后。
2. 修改 `hagrid_v2/run.py`：
   - 在实例化 `Trainer` 时，根据 `args.command` ('train' 或 'test') 传入对应的 `log_subdir`。

## 3. 预期结果
- 训练命令生成的日志 -> `.../logs/train/events...`
- 测试命令生成的日志 -> `.../logs/test/events...`
- 互不干扰，且 TensorBoard 能同时显示。

是否执行此计划？