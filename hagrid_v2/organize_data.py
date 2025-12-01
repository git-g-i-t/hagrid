# organize_data.py
import os
import shutil
import json
from omegaconf import OmegaConf

def organize_data_by_category(config_path):
    config = OmegaConf.load(config_path)
    
    for dataset_type in ['train', 'val', 'test']:
        print(f"\n整理 {dataset_type} 数据...")
        
        # 获取标注文件路径
        json_base_path = config.dataset.get(f"annotations_{dataset_type}")
        dataset_path = config.dataset.get(f"dataset_{dataset_type}")
        
        # 创建类别文件夹
        for target in config.dataset.targets:
            target_dir = os.path.join(dataset_path, target)
            os.makedirs(target_dir, exist_ok=True)
        
        # 读取标注并移动文件
        for target in config.dataset.targets:
            json_path = os.path.join(json_base_path, f"{target}.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
                
                # 移动图片到对应类别文件夹
                for image_name, annotation in annotations.items():
                    src_path = os.path.join(dataset_path, f"{image_name}.jpg")
                    dst_path = os.path.join(dataset_path, target, f"{image_name}.jpg")
                    
                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)
                        print(f"移动: {src_path} -> {dst_path}")
                    else:
                        print(f"警告: 找不到图片 {src_path}")

if __name__ == "__main__":
    organize_data_by_category("configs/convnext_base.yaml")