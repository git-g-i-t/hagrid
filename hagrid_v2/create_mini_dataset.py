import os
import json
import random
from PIL import Image  # 新增：用于图像处理
from tqdm import tqdm

def process_and_crop_image(src_path, dest_dir, bboxes, image_id):
    """
    读取原图，根据bbox裁剪出手部区域并保存
    注意：一张图里可能有多只手
    """
    try:
        with Image.open(src_path) as img:
            width, height = img.size
            
            # 遍历该图中的所有边界框
            for idx, box in enumerate(bboxes):
                # HaGRID bbox 格式通常是归一化的 [x, y, w, h] (0~1之间)
                # 需要转换成像素坐标
                x_norm, y_norm, w_norm, h_norm = box
                
                # 转换为绝对坐标
                x = int(x_norm * width)
                y = int(y_norm * height)
                w = int(w_norm * width)
                h = int(h_norm * height)
                
                # 增加一点边缘 (Padding)，比如扩充 10%，防止切太死
                pad_w = int(w * 0.1)
                pad_h = int(h * 0.1)
                
                left = max(0, x - pad_w)
                top = max(0, y - pad_h)
                right = min(width, x + w + pad_w)
                bottom = min(height, y + h + pad_h)
                
                # 裁剪
                crop_img = img.crop((left, top, right, bottom))
                
                # 保存文件名需加上索引，因为一张图可能有多个手势
                # 例如: original_id_0.jpg
                save_name = f"{image_id}_{idx}.jpg"
                crop_img.save(os.path.join(dest_dir, save_name), quality=95)
                
        return True
    except Exception as e:
        print(f"裁剪失败 {src_path}: {e}")
        return False

def create_mini_dataset(annotations_dir, dataset_dir, output_dataset_dir, targets, max_samples_per_class=2000):
    """
    修改版：不再复制 JSON，而是直接生成裁剪后的图片数据集结构。
    这种结构 (ImageFolder 格式) 可以直接被 PyTorch/TensorFlow 读取。
    结构示例:
       dataset_mini/
          train/
             call/
                img1.jpg
             fist/
                ...
    """
    
    phases = ['train', 'val'] # 简化，先做训练和验证
    
    for phase in phases:
        print(f"\n开始处理阶段: {phase}")
        phase_anno_dir = os.path.join(annotations_dir, phase)
        
        # 遍历目标类别
        for target in targets:
            json_file = f"{target}.json"
            json_path = os.path.join(phase_anno_dir, json_file)
            
            if not os.path.exists(json_path):
                print(f"跳过: 找不到 {json_path}")
                continue
                
            # 创建输出目录：dataset_mini/train/call
            target_out_dir = os.path.join(output_dataset_dir, phase, target)
            os.makedirs(target_out_dir, exist_ok=True)
            
            # 读取标注
            print(f"正在加载 {target} 的标注...")
            with open(json_path, 'r') as f:
                data = json.load(f) # data结构: {"img_id": {"bboxes": [[x,y,w,h]], "labels": ["call"]}}
            
            image_ids = list(data.keys())
            
            # 随机采样：与其按比例，不如固定数量更稳妥（防止某些类样本过少）
            if len(image_ids) > max_samples_per_class:
                sampled_ids = random.sample(image_ids, max_samples_per_class)
            else:
                sampled_ids = image_ids
                
            count = 0
            for image_id in tqdm(sampled_ids, desc=f"裁剪 {target}"):
                # 查找原图路径
                src_image_path = find_image_path(image_id, os.path.join(dataset_dir, target))
                
                if src_image_path:
                    # 获取该图的 bboxes
                    # 注意：HaGRID json 里的 key 对应的值里包含 bboxes 列表
                    item_info = data[image_id]
                    bboxes = item_info.get('bboxes', [])
                    
                    # 执行裁剪并保存
                    success = process_and_crop_image(src_image_path, target_out_dir, bboxes, image_id)
                    if success:
                        count += 1
            
            print(f"类别 {target} 处理完成，生成图片约 {count} 张")

# 辅助函数保持不变
def find_image_path(image_id, category_dir):
    extensions = ['.jpg', '.jpeg', '.png']
    if not os.path.exists(category_dir):
        return None 
    for ext in extensions:
        p = os.path.join(category_dir, image_id + ext)
        if os.path.exists(p): return p
    return None

if __name__ == "__main__":
    # 配置你的路径
    # 注意：HaGRID 下载后的图片通常是分 zip 包的，确保解压后的 dataset_dir 结构正确
    anno_dir = "./hagrid_dataset/annotations" 
    data_dir = "./hagrid_dataset/images"      
    out_dir = "./my_hagrid_mini"
    
    # 建议先只跑这几个简单的类进行测试
    my_targets = ["call", "fist", "like", "ok", "palm", "stop"]
    
    create_mini_dataset(anno_dir, data_dir, out_dir, my_targets, max_samples_per_class=1000)