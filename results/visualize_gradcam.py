import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from hagrid_v2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 尝试导入 grad-cam 库
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("❌ 错误: 未找到 'grad-cam' 库。")
    print("请运行: pip install grad-cam")
    exit(1)

# 导入你的模型构建函数
from hagrid_v2.custom_utils.utils import build_model

def get_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--config", type=str, default="hagrid_v2/configs/se_resnet18.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image_dir", type=str, default="hagrid_v2/dataset_mini/test", help="Directory with test images")
    parser.add_argument("--output_dir", type=str, default="results/gradcam", help="Directory to save results")
    parser.add_argument("--target_layer", type=str, default="layer4", help="Target layer for Grad-CAM (e.g., layer4)")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to visualize")
    return parser.parse_args()

def preprocess_image(img_path, img_size=224):
    """
    读取并预处理图片
    """
    # 读取原始图片用于显示
    rgb_img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    rgb_img = cv2.resize(rgb_img, (img_size, img_size))
    rgb_img_float = np.float32(rgb_img) / 255.0 # 归一化到 [0, 1] 用于 grad-cam

    # 预处理用于模型输入
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.54, 0.499, 0.474], std=[0.234, 0.235, 0.231])
    ])
    
    input_tensor = transform(rgb_img).unsqueeze(0) # (1, C, H, W)
    return rgb_img_float, input_tensor

def main():
    args = get_args()
    
    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 加载配置和模型
    print(f"Loading config from {args.config}...")
    conf = OmegaConf.load(args.config)
    
    print(f"Building model {conf.model.name}...")
    # 这里我们要临时修改配置里的 pretrained 为 False，因为我们加载的是本地 checkpoint
    conf.model.pretrained = False 
    model = build_model(conf)
    
    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理可能的 state_dict key 不匹配问题 (比如带了 "module." 前缀)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        # 还要去掉 hagrid_model 前缀 (因为 ClassifierModel 包装了一层)
        name = name.replace("hagrid_model.", "") if name.startswith("hagrid_model.") else name
        new_state_dict[name] = v
        
    # 由于我们的模型被 ClassifierModel 包装了，我们需要把权重加载到内部的 hagrid_model
    try:
        model.hagrid_model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print(f"⚠️ 权重加载部分不匹配 (可能是分类头维度不同)，但这通常不影响 Grad-CAM 可视化主干网络: {e}")

    model.to(device)
    model.eval()

    # 3. 设置 Grad-CAM 目标层
    # 对于 ResNet，通常是 layer4 (最后一个卷积层)
    target_layers = [getattr(model.hagrid_model, args.target_layer)[-1]]
    
    cam = GradCAM(model=model.hagrid_model, target_layers=target_layers) # use_cuda=True if device.type=='cuda' else False

    # 4. 遍历图片并生成热力图
    # 递归查找图片
    image_paths = []
    for root, dirs, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= args.num_images:
                    break
        if len(image_paths) >= args.num_images:
            break
            
    if not image_paths:
        print(f"❌ 在 {args.image_dir} 下没找到图片！")
        return

    print(f"Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        try:
            filename = os.path.basename(img_path)
            print(f"[{i+1}/{len(image_paths)}] Processing {filename}...")
            
            # 预处理
            rgb_img, input_tensor = preprocess_image(img_path, img_size=conf.dataset.img_size)
            input_tensor = input_tensor.to(device)
            
            # 模型预测 (获取预测类别)
            with torch.no_grad():
                output = model.hagrid_model(input_tensor)
                pred_idx = output.argmax(dim=1).item()
                conf_score = output.softmax(dim=1).max().item()
            
            # 生成 Grad-CAM
            # targets=None 表示自动选择概率最高的类别作为目标
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # 叠加热力图
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # 绘图保存
            plt.figure(figsize=(10, 5))
            
            # 左图：原图
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_img)
            plt.title(f"Original: {filename}")
            plt.axis('off')
            
            # 右图：Grad-CAM
            plt.subplot(1, 2, 2)
            plt.imshow(visualization)
            plt.title(f"Grad-CAM (Pred: {pred_idx}, Conf: {conf_score:.2f})")
            plt.axis('off')
            
            save_path = os.path.join(args.output_dir, f"gradcam_{filename}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"✅ Saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

    print("\n🎉 All Done! Check results in 'results/gradcam' folder.")

if __name__ == "__main__":
    main()
