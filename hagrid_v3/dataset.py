import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mindspore as ms # 替代 torch
from omegaconf import DictConfig
from PIL import Image
# PyTorch 的 Dataset 基类移除，MindSpore 只需要类实现 __getitem__ 和 __len__
from tqdm import tqdm

# 导入支持的图片扩展名
from constants import IMAGES

class HagridDataset: # 移除了 Dataset 继承
    """
    HaGRID 数据集的基类
    负责初始化配置、读取 JSON 标注文件、过滤无效图片等通用操作。
    """

    def __init__(self, conf: DictConfig, dataset_type: str, transform):
        """
        Parameters
        ----------
        conf : DictConfig
            配置对象 (来自 config/*.yaml)
        dataset_type : str
            数据集类型: "train", "val" 或 "test"
        transform : albumentations.Compose
            数据增强和预处理管道
        """
        self.conf = conf
        # --- 路径自动化处理开始 ---
        # 获取当前脚本所在目录的上一级，即项目根目录 F:/hagrid
        # __file__ 是 dataset.py 的路径，dirname 是 hagrid_v3，再 dirname 就是 HAGRID 根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 获取配置中的相对路径
        rel_path_json = self.conf.dataset.get(f"annotations_{dataset_type}")
        rel_path_data = self.conf.dataset.get(f"dataset_{dataset_type}")

        # 动态拼接成绝对路径
        self.path_to_json = os.path.join(project_root, rel_path_json)
        self.path_to_dataset = os.path.join(project_root, rel_path_data)
        # --- 路径自动化处理结束 ---
        # 构建类别映射字典：例如 {'call': 0, 'like': 1, ...}
        # 这对于将字符串标签转换为模型需要的数字 ID 至关重要
        self.labels = {
            label: num for (label, num) in zip(self.conf.dataset.targets, range(len(self.conf.dataset.targets)))
        }

        self.dataset_type = dataset_type

        # 数据子集大小：如果是训练集且配置了 subset，则只取前 N 张（用于快速调试）
        # -1 表示使用全部数据
        subset = self.conf.dataset.get("subset", None) if dataset_type == "train" else -1

        # 获取 JSON 标注文件夹路径 和 图片文件夹路径
        self.path_to_json = os.path.expanduser(self.conf.dataset.get(f"annotations_{dataset_type}"))
        self.path_to_dataset = os.path.expanduser(self.conf.dataset.get(f"dataset_{dataset_type}"))
        
        # 核心步骤：读取所有 JSON 并生成 Pandas DataFrame
        self.annotations = self.__read_annotations(subset)

        self.transform = transform

    @staticmethod
    def _load_image(image_path: str):
        """
        使用 PIL 读取图片并转为 RGB 格式
        """
        image = Image.open(image_path).convert("RGB")
        return image

    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple) -> List:
        """
        获取指定目录下的所有有效图片文件名
        """
        if not os.path.exists(pth):
            logging.warning(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        return files

    def __read_annotations(self, subset: int = None) -> pd.DataFrame:
        """
        读取并解析 JSON 标注文件，生成总的数据表
        
        逻辑：
        1. 遍历所有手势类别 (targets)。
        2. 读取该类别对应的 JSON 文件。
        3. 将 JSON 转换为 DataFrame。
        4. 扫描硬盘，确认哪些图片是真实存在的。
        5. 只保留【标注存在 且 图片存在】的数据行。
        """
        exists_images = set() # 使用集合(set)存储存在的文件名，查询速度更快
        annotations_all = []

        for target in tqdm(self.conf.dataset.targets, desc=f"Prepare {self.dataset_type} dataset"):
            target_tsv = os.path.join(self.path_to_json, f"{target}.json")
            if os.path.exists(target_tsv):
                with open(target_tsv, "r") as file:
                    json_annotation = json.load(file)

                # 将 JSON 字典展平为列表，并添加 .jpg 后缀
                json_annotation = [
                    {**annotation, "name": f"{name}.jpg"} for name, annotation in json_annotation.items()
                ]
                
                # 如果设置了 subset，只取前一部分数据
                if subset > 1:
                    json_annotation = json_annotation[:subset]

                annotation = pd.DataFrame(json_annotation)
                annotation["target"] = target
                annotations_all.append(annotation)
                
                # 扫描该类别下的实际图片文件
                exists_images.update(self.__get_files_from_dir(os.path.join(self.path_to_dataset, target), IMAGES))
            else:
                logging.info(f"Database for {target} not found")

        # 合并所有类别的 DataFrame
        annotations_all = pd.concat(annotations_all, ignore_index=True)
        # 标记哪些行对应的图片文件真实存在
        annotations_all["exists"] = annotations_all["name"].isin(exists_images)
        # 返回过滤后的数据表
        return annotations_all[annotations_all["exists"]]

    def __getitem__(self, item):
        # 基类不实现具体取数据逻辑，由子类实现
        raise NotImplementedError

    def __len__(self):
        # 返回数据集总样本数
        return self.annotations.shape[0]


class ClassificationDataset(HagridDataset):
    """
    图像分类专用数据集 (用于训练 ResNet, MobileNet 等)
    返回: Image, Label
    """
    def __init__(self, conf: DictConfig, dataset_type: str, transform):
        super().__init__(conf, dataset_type, transform)
        
        # --- 数据清洗 ---
        # 过滤掉那些标签含糊不清的数据
        # 逻辑：如果一张图的 labels 列表里只有 ["no_gesture"]，但它的文件夹归类 (target) 却不是 "no_gesture"
        # 说明这张图可能标注错了或者手势不明显，直接扔掉不训练
        self.annotations = self.annotations[
            ~self.annotations.apply(lambda x: x["labels"] == ["no_gesture"] and x["target"] != "no_gesture", axis=1)
        ]
        self.dataset_type = dataset_type

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict]:
        """
        获取单条分类数据
        """
        row = self.annotations.iloc[[index]].to_dict("records")[0]
        image_pth = os.path.join(self.path_to_dataset, row["target"], row["name"])
        image = self._load_image(image_pth)

        labels = row["labels"]

        # --- 确定唯一标签 ---
        # 分类任务每张图只能有一个 Label，但 HaGRID 原生标注可能有多个
        if row["target"] == "no_gesture":
            gesture = "no_gesture"
        else:
            # 优先取第一个不是 "no_gesture" 的标签作为该图的分类真值
            for label in labels:
                if label == "no_gesture":
                    continue
                else:
                    gesture = label
                    break
        
        try:
            # 迁移点：torch.tensor -> np.array
            # 保持字典输出逻辑不变
            label = {"labels": np.array(self.labels[gesture], dtype=np.int32)}
        except Exception:
            raise f"unknown gesture {gesture}"
            
        # --- 数据增强 ---
        image = np.array(image)
        if self.transform is not None:
            # 分类任务只需要变换 Image，不需要管 BBox
            # 这里调用 albumentations，由于环境已好，逻辑完全不需要动
            image = self.transform(image=image)["image"]
            
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, label