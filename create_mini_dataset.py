import os
import json
import shutil
import random
from tqdm import tqdm

def create_mini_dataset(annotations_dir, dataset_dir, output_annotations_dir, output_dataset_dir, targets, sample_ratio=0.01):
    """
    创建缩小版数据集，每类抽取指定比例的数据

    参数:
    annotations_dir: 原始标注文件目录
    dataset_dir: 原始数据集目录
    output_annotations_dir: 输出标注文件目录
    output_dataset_dir: 输出数据集目录
    targets: 目标类别列表
    sample_ratio: 采样比例，默认为1%
    """

    # 阶段列表
    phases = ['train', 'val', 'test']

    # 创建输出目录
    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_dataset_dir, exist_ok=True)

    # 为每个阶段创建子目录
    for phase in phases:
        os.makedirs(os.path.join(output_annotations_dir, phase), exist_ok=True)
        os.makedirs(os.path.join(output_dataset_dir, phase), exist_ok=True)

    # 统计信息
    stats = {}

    # 遍历每个阶段
    for phase in phases:
        phase_dir = os.path.join(annotations_dir, phase)
        output_phase_dir = os.path.join(output_annotations_dir, phase)
        output_dataset_phase_dir = os.path.join(output_dataset_dir, phase)

        if not os.path.exists(phase_dir):
            print(f"阶段目录不存在: {phase_dir}")
            continue

        stats[phase] = {}

        # 遍历每个类别
        for target in tqdm(targets, desc=f"处理{phase}阶段"):
            json_file = f"{target}.json"
            json_path = os.path.join(phase_dir, json_file)
            output_json_path = os.path.join(output_phase_dir, json_file)

            # 检查JSON文件是否存在
            if not os.path.exists(json_path):
                print(f"标注文件不存在: {json_path}")
                continue

            # 读取JSON文件
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 获取所有图片ID
                image_ids = list(data.keys())
                total_images = len(image_ids)

                # 计算采样数量（至少1张）
                sample_size = max(1, int(total_images * sample_ratio))

                # 随机采样
                sampled_ids = random.sample(image_ids, sample_size)

                # 创建新的标注数据
                new_data = {}
                for image_id in sampled_ids:
                    new_data[image_id] = data[image_id]

                # 保存新的标注文件
                with open(output_json_path, 'w') as f:
                    json.dump(new_data, f, indent=2)

                # 复制对应的图片文件
                for image_id in sampled_ids:
                    # 查找图片文件
                    image_path = find_image_path(image_id, os.path.join(dataset_dir, target))
                    if image_path:
                        # 创建目标目录
                        target_dir = os.path.join(output_dataset_phase_dir, target)
                        os.makedirs(target_dir, exist_ok=True)

                        # 复制图片
                        shutil.copy2(image_path, os.path.join(target_dir, os.path.basename(image_path)))

                # 记录统计信息
                stats[phase][target] = {
                    'total': total_images,
                    'sampled': sample_size,
                    'ratio': sample_size / total_images if total_images > 0 else 0
                }

                print(f"完成: {phase}/{target} - 总数: {total_images}, 采样: {sample_size}")

            except Exception as e:
                print(f"处理标注文件失败 {json_path}: {e}")

    # 输出统计信息
    print("\n" + "="*60)
    print("数据集缩小完成")
    print("="*60)

    for phase in phases:
        if phase in stats:
            print(f"\n{phase}阶段统计:")
            for target, info in stats[phase].items():
                print(f"  {target}: {info['sampled']}/{info['total']} ({info['ratio']:.2%})")

def find_image_path(image_id, category_dir):
    """
    根据图片ID在指定类别目录中查找对应的图片文件路径

    参数:
    image_id: 图片ID（不带扩展名）
    category_dir: 类别目录

    返回:
    str: 图片文件路径，如果找不到则返回None
    """
    # 尝试不同的图片扩展名
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

    if not os.path.exists(category_dir):
        return None

    for ext in extensions:
        image_path = os.path.join(category_dir, image_id + ext)
        if os.path.exists(image_path):
            return image_path

    return None

def main():
    """主函数"""
    # 配置路径和目标类别
    annotations_dir = "annotations"  # 原始标注文件目录
    dataset_dir = "dataset"          # 原始数据集目录
    output_annotations_dir = "annotations_mini"  # 输出标注文件目录
    output_dataset_dir = "dataset_mini"          # 输出数据集目录

    # 目标类别列表
    targets = [
        "grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign",
        "hand_heart", "hand_heart2", "little_finger", "middle_finger", "take_picture",
        "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
        "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2",
        "two_up", "two_up_inverted", "three_gun", "thumb_index", "thumb_index2", "no_gesture"
    ]

    # 创建缩小版数据集
    create_mini_dataset(
        annotations_dir, 
        dataset_dir, 
        output_annotations_dir, 
        output_dataset_dir, 
        targets,
        sample_ratio=0.01  # 1%
    )

if __name__ == "__main__":
    main()