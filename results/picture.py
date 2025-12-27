import os
from PIL import Image

def process_gradcam_images(image_paths, output_path, spacing=10):
    """
    裁切并重新组合 Grad-CAM 图片
    布局: [Original] | [Heatmap 1] | [Heatmap 2] | [Heatmap 3]
    """
    parts = []
    
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        w, h = img.size
        mid = w // 2  # 假设原图和热力图各占一半宽度
        
        if i == 0:
            # 第一张图：同时保留左边的原图和右边的热力图
            original_part = img.crop((0, 0, mid, h))   # 左半部分
            heatmap_part = img.crop((mid, 0, w, h))    # 右半部分
            parts.append(original_part)
            parts.append(heatmap_part)
        else:
            # 后续图片：只保留右边的热力图
            heatmap_part = img.crop((mid, 0, w, h))
            parts.append(heatmap_part)

    # --- 统一所有部分的高度 (防止有像素级误差) ---
    target_height = parts[0].height
    resized_parts = []
    for p in parts:
        if p.height != target_height:
            new_w = int(p.width * (target_height / p.height))
            p = p.resize((new_w, target_height), Image.Resampling.LANCZOS)
        resized_parts.append(p)

    # --- 计算总宽度 ---
    total_width = sum(p.width for p in resized_parts) + (len(resized_parts) - 1) * spacing
    
    # --- 创建画布并拼接 ---
    new_img = Image.new('RGB', (total_width, target_height), (255, 255, 255))
    
    current_x = 0
    for p in resized_parts:
        new_img.paste(p, (current_x, 0))
        current_x += p.width + spacing

    # --- 保存 ---
    new_img.save(output_path, quality=95)
    print(f"✅ 处理完成！包含 1张原图 + {len(image_paths)}张热力图")
    print(f"📍 已保存至: {output_path}")

# ================= 配置区域 =================
# 顺序一定要对：[图1, 图2, 图3]
image_files = [
    "results\\gradcam\\ResNet18_pre\\gradcam_10889602-302a-4975-a9f3-be2beac38e21.jpg", 
    "results/gradcam/se_resnet18/gradcam_10889602-302a-4975-a9f3-be2beac38e21.jpg", 
    "results\\gradcam\\cbam_resnet18\\gradcam_10889602-302a-4975-a9f3-be2beac38e21.jpg"
]
output_name = "results\\final_combined_layout1.jpg"
# ===========================================

if __name__ == "__main__":
    # 确保文件存在
    valid_files = [f for f in image_files if os.path.exists(f)]
    if len(valid_files) < 3:
        print("❌ 错误：请确保文件夹下有三张原始 Grad-CAM 图片。")
    else:
        process_gradcam_images(valid_files, output_name, spacing=15)