import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from custom_utils.train_utils import (
    load_train_objects,
    load_train_optimizer,
    Trainer
)
from custom_utils.utils import (
    F1ScoreWithLogging,
    set_random_seed
)

from models.classifiers.se_resnet import SEResNet, SEBasicBlock
from models.classifiers.base_model_my import ClassifierModel

# ================= 配置区域 =================
CONFIG_PATH = "hagrid_v2/configs/se_resnet18.yaml"
SEARCH_EPOCHS = 25
GPU_ID = 0

# 搜索空间：分类头隐藏层结构
SEARCH_SPACE = {
    #"0-Layer (Standard)": [],
    #"1-Layer (Hidden 512)": [512],
    # "1-Layer (Hidden 256)": [256],
    # "2-Layers (512->256)": [512, 256],
    # "3-Layers (512->256->128)": [512, 256, 128]
}
# ===========================================


def run_search():
    set_random_seed(42)

    conf = OmegaConf.load(CONFIG_PATH)

    # 搜索阶段：短周期训练
    conf.epochs = SEARCH_EPOCHS
    conf.model.pretrained = False  # 明确从 0 开始训练

    # 1. 加载数据（只加载一次）
    print("📦 正在加载数据...")
    train_loader, val_loader, test_loader, _ = load_train_objects(
        conf, "train", n_gpu=1
    )

    results = {
        "0-Layer (Standard)": 0.83,
        "1-Layer (512)": 0.612,
        "1-Layer (256)": 0.556,
        "2-Layers": 0.667,
        "3-Layers": 0.500
    }

    # 2. 结构搜索
    for name, hidden_layers in SEARCH_SPACE.items():
        print(f"\n🚀 测试结构: {name} | hidden_layers={hidden_layers}")

        # ---------- 构建模型 ----------
        backbone = SEResNet(
            SEBasicBlock,
            [2, 2, 2, 2],
            num_classes=len(conf.dataset.targets),
            hidden_layers=hidden_layers,
        )

        model_wrapper = ClassifierModel(
            lambda **k: backbone,
            num_classes=len(conf.dataset.targets),
        )
        model_wrapper.type = "classifier"
        model_wrapper.criterion = getattr(torch.nn, conf.criterion)()

        # ---------- 优化器 & 指标 ----------
        optimizer, scheduler = load_train_optimizer(model_wrapper, conf)
        metric = F1ScoreWithLogging(
            task="multiclass",
            num_classes=len(conf.dataset.targets),
        )

        # 防止实验日志覆盖
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        conf.experiment_name = f"Search_{safe_name}"

        trainer = Trainer(
            model=model_wrapper,
            config=conf,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_calculator=metric,
            train_data=train_loader,
            val_data=val_loader,
            test_data=test_loader,
            n_gpu=1,
        )

        # ---------- 开始训练 ----------
        trainer.train()

        # ============================================================
        # ✅ 稳定性评价指标（关键修改点）
        # 取最后 K 个 epoch 的平均 F1，而不是 max
        # ============================================================
        K = 5

        if hasattr(trainer, "val_f1_history"):
            val_f1s = trainer.val_f1_history
        elif hasattr(trainer, "history") and \
             "val" in trainer.history and \
             "F1Score" in trainer.history["val"]:
            val_f1s = trainer.history["val"]["F1Score"]
        else:
            raise RuntimeError(
                "❌ Trainer 中未找到验证 F1 历史，请保存 val F1 记录"
            )

        if len(val_f1s) >= K:
            stable_f1 = sum(val_f1s[-K:]) / K
        else:
            stable_f1 = sum(val_f1s) / len(val_f1s)

        results[name] = stable_f1
        print(f"✅ {name} 稳定 F1 (last {K} epochs): {stable_f1:.4f}")

    # 3. 画图总结
    plot_results(results)


def plot_results(results):
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    scores = list(results.values())

    plt.bar(names, scores)
    plt.title(f"MLP Depth Search (Stable Metric, Epochs={SEARCH_EPOCHS})")
    plt.ylabel("Stable Validation F1")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/mlp_depth_search_stable.png")
    print("\n📊 搜索结果已保存为 results/mlp_depth_search_stable.png")
    plt.show()


if __name__ == "__main__":
    run_search()
