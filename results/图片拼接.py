import os
from PIL import Image

def concat_images_vertical(image_paths, output_path, spacing=15):
    """
    将三张图片进行上下垂直拼接。
    :param image_paths: 图片路径列表
    :param output_path: 输出路径
    :param spacing: 图片之间的间距（像素）
    """
    # 1. 打开所有图片
    images = [Image.open(x) for x in image_paths]
    
    # 2. 统一图片宽度（以第一张图的宽度为基准）
    target_width = images[0].width
    resized_images = []
    
    for img in images:
        if img.width != target_width:
            # 等比例计算缩放后的高度
            new_height = int(img.height * (target_width / img.width))
            img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)

    # 3. 计算总高度和最大宽度
    total_height = sum(img.height for img in resized_images) + (len(resized_images) - 1) * spacing
    max_width = target_width

    # 4. 创建白色背景的新画布
    new_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))

    # 5. 开始粘贴图片
    current_y = 0
    for img in resized_images:
        new_img.paste(img, (0, current_y))
        current_y += img.height + spacing

    # 6. 保存
    new_img.save(output_path, quality=95)
    print(f"✅ 上下拼接完成！已保存至: {output_path}")

# ================= 配置区域 =================
# 顺序为：[最上方图片, 中间图片, 最下方图片]
image_files = [
    "results\\gradcam\\final\\final_combined_layout1.jpg", 
    "results\\gradcam\\final\\final_combined_layout4.jpg", 
    "results\\gradcam\\final\\final_combined_layout5.jpg"
]
output_name = "results\\gradcam\\final\\stacked_result_vertical.jpg"
# ===========================================

if __name__ == "__main__":
    # 检查文件
    valid_files = [f for f in image_files if os.path.exists(f)]
    if len(valid_files) < 3:
        print("❌ 错误：请检查图片路径是否正确，需要三张图片。")
    else:
        concat_images_vertical(valid_files, output_name, spacing=20)