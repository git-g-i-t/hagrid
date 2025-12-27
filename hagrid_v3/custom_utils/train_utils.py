import os
from typing import List, Tuple, Dict

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.dataset as ds
from omegaconf import DictConfig
# 使用 tensorboardX 替代 torch.utils.tensorboard，彻底移除 torch 依赖
from tensorboardX import SummaryWriter

# 导入你迁移好的组件
from custom_utils.utils import Logger, build_model, get_transform

def get_available_device():
    """
    MindSpore CPU 版固定返回 CPU
    """
    return "CPU"

def collate_fn(batch: List) -> Tuple:
    """
    保持函数名和逻辑不变。
    """
    return list(zip(*batch))

def get_dataloader(dataset, **kwargs) -> ds.GeneratorDataset:
    """
    构建 MindSpore 数据加载管道
    """
    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["image", "label"],
        shuffle=kwargs.get("shuffle", False),
        num_parallel_workers=kwargs.get("num_workers", 1) if kwargs.get("num_workers", 1) > 0 else 1
    )
    
    # 设置 Batch 大小，MindSpore 会自动处理 Tensor 的堆叠 (Stack)
    dataloader = dataloader.batch(batch_size=kwargs["batch_size"])
    return dataloader

def load_train_objects(config: DictConfig, command: str):
    """
    核心工厂函数：加载训练所需的所有对象
    """
    model = build_model(config)

    from dataset import ClassificationDataset as GestureDataset

    # 初始化测试集
    test_dataset = GestureDataset(config, "test", get_transform(config.test_transforms, model.type))
    test_dataloader = get_dataloader(test_dataset, **config.test_params)

    # 如果是训练模式，初始化训练集和验证集
    if command == "train":
        train_dataset = GestureDataset(config, "train", get_transform(config.train_transforms, model.type))
        train_dataloader = get_dataloader(train_dataset, **config.train_params)
        
        if config.dataset.dataset_val and config.dataset.annotations_val:
            val_dataset = GestureDataset(config, "val", get_transform(config.val_transforms, model.type))
            val_dataloader = get_dataloader(val_dataset, **config.val_params)
        else:
            val_dataloader = None
    else:
        train_dataloader = None
        val_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader, model

def load_train_optimizer(model, config: DictConfig):
    """
    加载 MindSpore 优化器 (处理 AdamW 映射)
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    
    opt_name = config.optimizer.name
    
    # 适配 MindSpore 的 AdamW 名称
    if opt_name == "AdamW":
        optimizer_cls = nn.AdamWeightDecay
    else:
        optimizer_cls = getattr(nn, opt_name)
        
    optimizer = optimizer_cls(parameters, **config.optimizer.params)
    
    scheduler = None
    return optimizer, scheduler

class Trainer:
    """
    训练总管 (MindSpore CPU 适配版)
    """
    def __init__(
        self,
        model,
        config: DictConfig,
        test_data: ds.GeneratorDataset,
        train_data: ds.GeneratorDataset = None,
        val_data: ds.GeneratorDataset = None,
        metric_calculator=None,
        optimizer=None,
        scheduler=None,
        log_subdir: str = None,
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        
        self.device = "CPU"
        self.model = model

        if self.model.type == "classifier":
            self.metric_name = "F1Score"
        else:
            self.metric_name = "map"
            
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.current_state = {"loss": 0, "metric": {self.metric_name: 0}, "epoch": 0}
        self.best_state = {"loss": 0, "metric": {self.metric_name: 0}, "epoch": 0}

        self.stop = False
        self.max_epoch = self.config.epochs
        self.epochs_run = 0
        self.metric_calculator = metric_calculator
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 动态拼接工作目录
        rel_work_dir = self.config.work_dir
        full_work_dir = os.path.join(project_root, rel_work_dir)
        # 初始化日志
        if not os.path.exists(full_work_dir):
            os.makedirs(full_work_dir)

        log_path = os.path.join(full_work_dir, self.config.experiment_name, "logs")
        if log_subdir:
            log_path = os.path.join(log_path, log_subdir)
            
        self.summary_writer = SummaryWriter(log_dir=log_path)

        if self.config.model.checkpoint is not None:
            self._load_snapshot(self.config.model.checkpoint)

        # --- 核心：定义梯度计算函数 ---
        def forward_fn(data, label):
            logits = self.model.hagrid_model(data)
            loss = self.model.criterion(logits, label)
            return loss, logits

        self.grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)

    def train_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss

    def _save_snapshot(self):
        metric_score = self.best_state["metric"][self.metric_name]
        save_path = os.path.join(self.config.work_dir, self.config.experiment_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_name = f"{self.config.model.name}_epoch-{self.best_state['epoch']}_{self.metric_name}-{metric_score:.2f}_loss-{self.best_state['loss']:.2f}.ckpt"
        ms.save_checkpoint(self.model.hagrid_model, os.path.join(save_path, save_name))
        print(f"Save model {self.config.model.name} || {self.metric_name}:{metric_score:.2f}")

    def test(self):
        """
        测试循环
        """
        self.model.eval()
        if self.test_data is None: raise Exception("Cannot test without test data")

        # 1. 清理数据
        self.metric_calculator.clear_accumulated()

        dataset_size = self.test_data.get_dataset_size()
        with Logger("Test", self.max_epoch, dataset_size, self.config.log_every, self.device) as logger:
            for iteration, data in enumerate(self.test_data.create_dict_iterator()):
                images = data["image"]
                targets_raw = data["label"]["labels"]

                output_dict = self.model(images)
                targets_list = [{"labels": t} for t in targets_raw]

                # 这里的计算器现在会返回 Accuracy 并在内部累积数据
                metric = self.metric_calculator(output_dict, targets_list)
                logger.log_iteration(iteration + 1, self.current_state["epoch"], metrics=metric)

            # 2. 获取并打印真正的 F1 和矩阵
            final_metrics = self.metric_calculator.print_final_matrix(self.current_state["epoch"], stage="Test")

            # 3. 记录到 TensorBoard（使用真正的 F1）
            if "F1Score" in final_metrics:
                self.summary_writer.add_scalar("F1Score/Test", final_metrics["F1Score"], self.current_state["epoch"])

    def val(self):
        """
        验证循环：每个 Epoch 结束后运行，确保模型保存基于全量真实的 F1-score
        """
        self.model.eval()
        
        # 1. 验证开始前：清理之前 Batch 累积的数据
        self.metric_calculator.clear_accumulated()

        dataset_size = self.val_data.get_dataset_size()
        with Logger("Eval", self.max_epoch, dataset_size, self.config.log_every, self.device) as logger:
            for iteration, data in enumerate(self.val_data.create_dict_iterator()):
                images = data["image"]
                targets_raw = data["label"]["labels"]

                output_dict = self.model(images)
                targets_list = [{"labels": t} for t in targets_raw]

                # 这里会把预测结果存入 metric_calculator.all_preds
                metric = self.metric_calculator(output_dict, targets_list)
                logger.log_iteration(iteration + 1, self.current_state["epoch"], metrics=metric)

            # 2. 关键点：获取整个验证集跑完后的【真实全量 F1】
            # 这个函数现在会返回 {"F1Score": 0.58xxx}
            final_metrics = self.metric_calculator.print_final_matrix(self.current_state["epoch"], stage="Eval")

            # 3. 强制更新 current_state，确保“保存逻辑”拿到的是准确的分数
            if "F1Score" in final_metrics:
                self.current_state["metric"]["F1Score"] = final_metrics["F1Score"]
            
            # 记录到 TensorBoard (使用真实分数)
            self.summary_writer.add_scalar(f"F1Score/Eval", self.current_state["metric"]["F1Score"], self.current_state["epoch"])

            # 4. 核心保存逻辑：使用修正后的 F1Score 进行比较
            current_f1 = self.current_state["metric"][self.metric_name]
            best_f1 = self.best_state["metric"][self.metric_name]

            if (current_f1 - best_f1) > self.config.early_stopping.metric:
                print(f"性能提升！F1 从 {best_f1:.4f} 提升至 {current_f1:.4f}，正在保存模型...")
                self.best_state.update({
                    "metric": self.current_state["metric"].copy(), 
                    "loss": self.current_state["loss"], 
                    "epoch": self.current_state["epoch"]
                })
                self._save_snapshot()

    def early_stop(self):
        return (self.current_state["epoch"] - self.best_state["epoch"] >= self.config.early_stopping.epochs)

    def train(self):
        if self.train_data is None: raise Exception("Cannot train without training data")
        
        for epoch in range(self.epochs_run, self.max_epoch):
            if self.early_stop(): self.stop = True
            if self.stop: break

            self.model.train()
            self.current_state["epoch"] = epoch
                
            dataset_size = self.train_data.get_dataset_size()
            with Logger("Train", self.max_epoch, dataset_size, self.config.log_every, self.device) as logger:
                for iteration, data in enumerate(self.train_data.create_dict_iterator()):
                    images = data["image"]
                    targets = data["label"]["labels"]

                    loss = self.train_step(images, targets)
                    logger.log_iteration(iteration + 1, epoch, loss.asnumpy().item())

                self.current_state["loss"] = logger.loss_averager.value
                self.summary_writer.add_scalar("loss/Train", self.current_state["loss"], epoch)

            if self.config.eval_every > 0 and epoch % self.config.eval_every == 0:
                self.val()

    def _load_snapshot(self, snapshot_path):
        """
        加载 MindSpore 权重文件 (.ckpt) 并检查匹配情况
        """
        if not os.path.exists(snapshot_path):
            print(f"警告: 找不到权重文件 {snapshot_path}，将从随机初始化开始。")
            return

        print(f"正在加载 MindSpore 权重: {snapshot_path}")
        
        # 1. 加载参数字典
        param_dict = ms.load_checkpoint(snapshot_path)
        
        # 2. 将参数加载到网络中
        # load_param_into_net 会返回两个列表，记录哪些没对上
        not_loaded, not_in_ckpt = ms.load_param_into_net(self.model.hagrid_model, param_dict)
        
        # 3. 打印检查结果 (这是判断模型是否正常加载的关键)
        if len(not_loaded) == 0:
            print("[加载成功] 模型所有参数已正确填充。")
        else:
            print(f"[部分加载] 模型中有 {len(not_loaded)} 个参数未被加载 (可能是分类头):")
            print(f"   未加载列表: {not_loaded[:5]} ...")

        if len(not_in_ckpt) > 0:
            print(f"ℹ[额外信息] 权重文件中多出了 {len(not_in_ckpt)} 个参数，已忽略。")