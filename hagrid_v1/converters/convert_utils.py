#==========================================
# 这份代码是 converters 模块的核心工具库。
# 它的主要作用是读取原始的 JSON 标注文件，并将其与实际存在的图片文件进行比对，最后清洗整合成一个统一的 Pandas DataFrame 表格。
# 这样做的好处是，无论后续要转成 YOLO 还是 COCO 格式，都可以直接从这个 DataFrame 里取数据，而不用每次都去解析复杂的 JSON 结构。
#===========================================
import json
import logging
import os
from typing import Union

import pandas as pd
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

# 从常量文件中导入支持的图片扩展名 (如 .jpg, .jpeg, .png)
from constants import IMAGES


def get_files_from_dir(pth, extns):
    """
    从指定目录获取符合扩展名的所有文件列表
    
    Parameters
    ----------
    pth: str
        目录路径
    extns: Tuple
        允许的文件扩展名元组，例如 ('.jpg', '.png')
    
    Returns
    -------
    list
        文件名列表
    """
    # 安全检查：如果目录不存在，记录错误并返回空列表
    if not os.path.exists(pth):
        logging.error(f"Dataset directory doesn't exist {pth}")
        return []
    
    # 列表推导式：遍历目录，保留符合后缀的文件
    files = [f for f in os.listdir(pth) if f.endswith(extns)]
    return files



def get_dataframe(conf: Union[DictConfig, ListConfig], phase: str) -> pd.DataFrame:
    """
    核心函数：读取标注并生成清洗后的 Pandas DataFrame
    
    逻辑：
    1. 遍历所有手势类别 (Targets)。
    2. 读取每个类别对应的 JSON 标注文件。
    3. 将 JSON 数据转换为 DataFrame 格式。
    4. 检查图片文件是否真的存在于磁盘上 (避免标注还在，但图片没下载或已删除的情况)。
    5. 过滤掉不存在的图片，返回干净的数据表。

    Parameters
    ----------
    conf: Union[DictConfig, ListConfig]
        配置对象，包含数据集路径、类别列表等信息
    phase: str
        数据集阶段，例如 "train", "val", "test"

    Returns
    -------
    pd.DataFrame
        包含所有有效标注信息的总表
    """
    # 从配置中提取路径和参数
    dataset_annotations = conf.dataset.dataset_annotations # 标注 JSON 的根目录
    dataset_folder = conf.dataset.dataset_folder           # 图片数据的根目录
    targets = conf.dataset.targets                         # 手势类别列表 (如 call, like, stop...)
    
    annotations_all = None  # 用于存储合并后的所有标注
    exists_images = []      # 用于存储所有实际存在于磁盘上的图片文件名

    # 使用 tqdm 显示进度条，遍历每一个手势类别
    for target in tqdm(targets):
        # 构造该类别、该阶段的 JSON 路径，例如: /data/annotations/train/call.json
        target_json = os.path.join(dataset_annotations, f"{phase}", f"{target}.json")
        
        if os.path.exists(target_json):
            # 加载 JSON 文件
            # HaGRID 的 JSON 结构通常是: {"image_id": {"bboxes": [...], "labels": [...]}, ...}
            json_annotation = json.load(open(os.path.join(target_json)))

            # 数据重构：
            # 原始 JSON 的 key 是图片 ID (无后缀)，value 是标注信息。
            # 这里通过 zip 将它们展平，并给 ID 加上 ".jpg" 后缀，作为新的 "name" 字段。
            # 结果变成一个字典列表: [{"name": "xxx.jpg", "bboxes": ...}, ...]
            json_annotation = [
                dict(annotation, **{"name": f"{name}.jpg"})
                for name, annotation in zip(json_annotation, json_annotation.values())
            ]

            # 将列表转换为 Pandas DataFrame
            annotation = pd.DataFrame(json_annotation)

            # 添加一列 "target"，标记这些数据属于哪个手势 (例如 "call")
            annotation["target"] = target
            
            # 将当前类别的 DataFrame 拼接到总表 annotations_all 中
            annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
            
            # 获取磁盘上该类别文件夹下实际存在的所有图片文件
            # 路径例如: /data/dataset/call/
            exists_images.extend(get_files_from_dir(os.path.join(dataset_folder, target), IMAGES))
        else:
            # 如果找不到对应的 JSON 文件，打印警告
            logging.warning(f"Database for {target} not found")

    # --- 数据清洗关键步骤 ---
    # 1. 判断 DataFrame 中的 "name" 是否存在于 exists_images (实际磁盘文件列表) 中
    # 2. 只有标注存在 且 图片文件也存在 的行，"exists" 列才为 True
    annotations_all["exists"] = annotations_all["name"].isin(exists_images)
    
    # 过滤掉那些实际上没有图片文件的标注行
    annotations = annotations_all[annotations_all["exists"]]

    # 返回清洗后的最终数据表
    return annotations

