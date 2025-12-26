import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from custom_utils.train_utils import load_train_objects, load_train_optimizer, Trainer
from custom_utils.utils import build_model, F1ScoreWithLogging, set_random_seed
from models.classifiers.se_resnet import SEResNet, SEBasicBlock
from models.classifiers.base_model_my import ClassifierModel

# ================= 配置区域 =================
CONFIG_PATH = "hagrid_v2/configs/se_resnet18.yaml"  # 基础配置文件
SEARCH_EPOCHS = 25                       # 每种结构跑多少轮 (不用跑太久，看趋势即可)
GPU_ID = 0

# 定义要搜索的结构 (隐藏层列表)
# []: 输入 -> 7 (原版 ResNet)
# [256]: 输入 -> 256 -> 7
# [512, 256]: 输入 -> 512 -> 256 -> 7
SEARCH_SPACE = {
    "0-Layer (Standard)": [],
    "1-Layer (Hidden 512)": [512],
    "1-Layer (Hidden 256)": [256],
    "2-Layers (512->256)": [512, 256],
    "3-Layers (512->256->128)": [512, 256, 128]
}
# ===========================================

def run_search():
    set_random_seed(42)
    conf = OmegaConf.load(CONFIG_PATH)
    
    # 强制修改配置以适应快速搜索
    conf.epochs = SEARCH_EPOCHS
    conf.model.pretrained = False # SE-Net 必须从头训练
    
    # 1. 加载数据 (只加载一次，节省时间)
    print("正在加载数据...")
    train_loader, val_loader, test_loader, _ = load_train_objects(conf, "train", n_gpu=1)
    
    # ❌ 原代码：results = {}
    
    # ✅ 修改后：手动填入前两个的结果 (请替换成你的真实分数)
    results = {}

    # 2. 开始循环搜索
    for name, hidden_layers in SEARCH_SPACE.items():
        print(f"\n🚀 开始测试结构: {name} | 隐藏层: {hidden_layers}")
        
        # --- 手动构建模型 ---
        # 实例化 SE-ResNet
        # 注意：这里我们绕过了 build_model，直接通过类实例化，为了传入 hidden_layers
        backbone = SEResNet(
            SEBasicBlock, [2, 2, 2, 2], 
            num_classes=len(conf.dataset.targets), 
            hidden_layers=hidden_layers
        )
        
        # 包装成 ClassifierModel (为了兼容 Trainer)
        # 我们这里用一个小技巧：创建一个伪造的构造函数 lambda
        model_wrapper = ClassifierModel(lambda **k: backbone, num_classes=len(conf.dataset.targets))

        model_wrapper.type = "classifier" 

        model_wrapper.criterion = getattr(torch.nn, conf.criterion)()
        
        # --- 准备训练器 ---
        optimizer, scheduler = load_train_optimizer(model_wrapper, conf)
        metric = F1ScoreWithLogging(task="multiclass", num_classes=len(conf.dataset.targets))
        
        # 临时修改实验名称，防止日志覆盖
        safe_name = name.replace(' ', '_').replace('->', 'to').replace('(', '').replace(')', '')
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
            n_gpu=1
        )
        
        # --- 开始训练 ---
        trainer.train()
        
        # 记录最佳 F1
        best_f1 = trainer.best_state["metric"]["F1Score"]
        results[name] = best_f1
        print(f"✅ {name} 完成! 最佳 F1: {best_f1:.4f}")

    # 3. 绘图总结
    plot_results(results)

def plot_results(results):
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    scores = list(results.values())
    
    # 柱状图
    bars = plt.bar(names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.title(f"Comparison of MLP Depth (Epochs={SEARCH_EPOCHS})", fontsize=14)
    plt.ylabel("Best Validation F1-Score")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("results/mlp_depth_search_result.png")
    print("\n📊 结果对比图已保存为: mlp_depth_search_result.png")
    plt.show()

if __name__ == "__main__":
    run_search()