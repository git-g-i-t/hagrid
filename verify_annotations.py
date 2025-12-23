import os
import json
import time

def find_image_in_new_structure(image_id, phase, target, dataset_dir):
    """
    在新的目录结构中查找图片文件
    
    参数:
    image_id: 图片ID（不带扩展名）
    phase: 阶段（train/val/test）
    target: 类别名称
    dataset_dir: 新的数据集目录
    
    返回:
    bool: 是否找到图片
    """
    # 尝试不同的图片扩展名
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    # 构建图片目录路径
    image_dir = os.path.join(dataset_dir, phase, target)
    
    if not os.path.exists(image_dir):
        return False
        
    for ext in extensions:
        image_path = os.path.join(image_dir, image_id + ext)
        if os.path.exists(image_path):
            return True
    
    return False

def verify_mini_dataset(annotations_dir, dataset_dir, targets):
    """
    验证缩小版数据集的标注文件与图片的对应关系
    
    参数:
    annotations_dir: 标注文件目录（新的结构）
    dataset_dir: 数据集目录（新的结构）
    targets: 目标类别列表
    """
    
    # 阶段列表
    phases = ['train', 'val', 'test']
    
    # 遍历每个阶段
    for phase in phases:
        phase_dir = os.path.join(annotations_dir, phase)
        if not os.path.exists(phase_dir):
            print(f"阶段目录不存在: {phase_dir}")
            continue
            
        # 遍历每个类别
        for target in targets:
            json_file = f"{target}.json"
            json_path = os.path.join(phase_dir, json_file)
            
            # 检查JSON文件是否存在
            if not os.path.exists(json_path):
                print(f"标注文件不存在: {json_path}")
                continue
                
            # 读取JSON文件
            try:
                print(f"开始验证: {phase}/{target}.json")
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # 逐个处理标注条目
                found_count = 0
                total_count = 0
                
                for image_id, annotation in data.items():
                    total_count += 1
                    
                    # 在新的目录结构中查找图片
                    found = find_image_in_new_structure(image_id, phase, target, dataset_dir)
                    
                    # 输出结果
                    print(f"{image_id}: {found}")
                    
                    if found:
                        found_count += 1
                
                # 输出该文件的验证总结
                print(f"验证完成: {phase}/{target}.json - 找到 {found_count}/{total_count} 张图片")
                
                # 处理完一个文件后停顿一秒
                time.sleep(1)
                        
            except Exception as e:
                print(f"处理标注文件失败 {json_path}: {e}")

def check_directory_structure(annotations_dir, dataset_dir, targets):
    """
    检查目录结构是否符合预期
    """
    print("检查目录结构...")
    
    phases = ['train', 'val', 'test']
    
    # 检查标注目录结构
    for phase in phases:
        phase_annot_dir = os.path.join(annotations_dir, phase)
        if not os.path.exists(phase_annot_dir):
            print(f"警告: 标注阶段目录不存在 - {phase_annot_dir}")
            continue
            
        for target in targets:
            json_file = f"{target}.json"
            json_path = os.path.join(phase_annot_dir, json_file)
            if not os.path.exists(json_path):
                print(f"警告: 标注文件不存在 - {json_path}")
    
    # 检查数据集目录结构
    for phase in phases:
        phase_data_dir = os.path.join(dataset_dir, phase)
        if not os.path.exists(phase_data_dir):
            print(f"警告: 数据集阶段目录不存在 - {phase_data_dir}")
            continue
            
        for target in targets:
            target_dir = os.path.join(phase_data_dir, target)
            if not os.path.exists(target_dir):
                print(f"警告: 数据集类别目录不存在 - {target_dir}")
            else:
                # 统计图片数量
                image_files = [f for f in os.listdir(target_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"  {phase}/{target}: {len(image_files)} 张图片")

def main():
    """主函数"""
    # 配置路径和目标类别
    annotations_dir = "annotations_mini"  # 新的标注文件目录
    dataset_dir = "dataset_mini"          # 新的数据集目录
    
    # 目标类别列表
    targets = [
        "grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign",
        "hand_heart", "hand_heart2", "little_finger", "middle_finger", "take_picture",
        "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
        "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2",
        "two_up", "two_up_inverted", "three_gun", "thumb_index", "thumb_index2", "no_gesture"
    ]
    
    # 首先检查目录结构
    check_directory_structure(annotations_dir, dataset_dir, targets)
    
    print("\n开始验证标注与图片的对应关系...")
    
    # 验证数据集
    verify_mini_dataset(annotations_dir, dataset_dir, targets)

if __name__ == "__main__":
    main()