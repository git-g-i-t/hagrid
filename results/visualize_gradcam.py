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

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (hagrid)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# hagrid_v2 目录
hagrid_v2_path = os.path.join(project_root, "hagrid_v2")

# 将这两个路径都添加到 sys.path
sys.path.append(project_root)
sys.path.append(hagrid_v2_path)
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
    # ========================================================
    # ✨ 在这里修改你的默认路径和参数 ✨
    # ========================================================
    DEFAULT_CONFIG = "hagrid_v2\\configs\\cbam_resnet18.yaml"      # 配置文件路径
    DEFAULT_CHECKPOINT = "hagrid_v2\\work_dir\\CBAM_ResNet18_Attention\\CBAM_ResNet18_epoch-29_F1Score-0.78_loss-0.44.pth"              # 权重文件路径
    DEFAULT_IMAGE_DIR = "hagrid_v2/dataset_mini/test"          # 测试图片文件夹
    DEFAULT_OUTPUT_DIR = "results/gradcam/cbam_resnet18"                    # 结果保存路径
    DEFAULT_TARGET_LAYER = "layer4"                           # 目标卷积层
    DEFAULT_NUM_IMAGES = 5                                    # 默认处理图片张数
    # ========================================================

    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
                        help=f"Path to config file (default: {DEFAULT_CONFIG})")
    
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, 
                        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})")
    
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, 
                        help=f"Directory with test images (default: {DEFAULT_IMAGE_DIR})")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    
    parser.add_argument("--target_layer", type=str, default=DEFAULT_TARGET_LAYER, 
                        help=f"Target layer for Grad-CAM (default: {DEFAULT_TARGET_LAYER})")
    
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES, 
                        help=f"Number of images to visualize (default: {DEFAULT_NUM_IMAGES})")
    
    return parser.parse_args()

def preprocess_image(img_path, img_size=224):
    """
    读取并预处理图片
    """
    # 读取原始图片用于显示
    raw_bgr = cv2.imread(img_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
        
    rgb_img = raw_bgr[:, :, ::-1] # BGR -> RGB
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
    
    # 检查关键文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误: 找不到权重文件 '{args.checkpoint}'，请检查 DEFAULT_CHECKPOINT 设置。")
        return

    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 加载配置和模型
    print(f"🚀 Loading config from {args.config}...")
    conf = OmegaConf.load(args.config)
    
    print(f"📦 Building model {conf.model.name}...")
    conf.model.pretrained = False 
    model = build_model(conf)
    
    # 加载权重
    print(f"💾 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        name = name.replace("hagrid_model.", "") if name.startswith("hagrid_model.") else name
        new_state_dict[name] = v
        
    try:
        model.hagrid_model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print(f"⚠️ 权重加载部分不匹配: {e}")

    model.to(device)
    model.eval()

    # 3. 设置 Grad-CAM 目标层
    try:
        target_layers = [getattr(model.hagrid_model, args.target_layer)[-1]]
        cam = GradCAM(model=model.hagrid_model, target_layers=target_layers)
    except Exception as e:
        print(f"❌ 目标层设置错误: {e}. 请检查 --target_layer 是否正确（如 'layer4'）。")
        return

    # 4. 遍历图片并生成热力图
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

    print(f"📸 Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        try:
            filename = os.path.basename(img_path)
            print(f"[{i+1}/{len(image_paths)}] Processing {filename}...")
            
            rgb_img, input_tensor = preprocess_image(img_path, img_size=conf.dataset.img_size)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model.hagrid_model(input_tensor)
                pred_idx = output.argmax(dim=1).item()
                conf_score = output.softmax(dim=1).max().item()
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_img)
            plt.title(f"Original: {filename}")
            plt.axis('off')
            
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

    print(f"\n🎉 All Done! Results saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()