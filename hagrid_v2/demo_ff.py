"""
全帧手势分类演示脚本
demo_ff.py是一个简化版的演示脚本
不进行手部位置检测（不画框），只简单粗暴地将摄像头捕获的整张画面直接送入分类模型，输出一个最可能的手势标签
如果想测试一个分类模型（比如 ResNet, MobileNet）的性能，或者你的应用场景是用户会把手放到摄像头正中央占满屏幕，就用这个。
如果需要在一个大场景中找到手在哪里并识别，请使用 demo.py
"""

#关于包的注释，详情请见demo.py
import argparse
import logging
import time
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        预处理函数：将 OpenCV 读取的图像转换为模型输入的 Tensor

        Parameters
        ----------
        img: np.ndarray
            摄像头的原始帧（BGR格式）
        transform :
            Albumentations 的变换流程（通常包含 Resize, Normalize 等）

        Returns
        -------
        transformed_image["image"]: Tensor
            处理好的张量，形状通常为 [C, H, W]
        """
        # 应用定义好的变换
        transformed_image = transform(image=img)
        #这里不需要返回图片的宽高尺寸，因为不需要像检测任务那样还原Box坐标
        return transformed_image["image"]

    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        """
        从配置文件动态构建预处理 Pipeline

        Parameters
        ----------
        transform_config: DictConfig
            config with test transforms
            配置中的 test_transforms 部分
        """
        # 根据配置文件的键值对，动态初始化 Albumentations 的转换类
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        # 最后添加转换为 Tensor 的步骤
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(classifier, transform) -> None:
        """
        主循环：捕获视频流并进行全帧分类

        Parameters
        ----------
        classifier : TorchVisionModel
            加载好的分类模型（注意：这里是 Classifier，不是 Detector）
        transform :
            albumentation transforms
            预处理变换
        """
        # 打开默认摄像头 (ID 0)
        cap = cv2.VideoCapture(0)
        # 初始化时间变量用于计算 FPS
        t1 = cnt = 0

        while cap.isOpened():
            # 计算两帧之间的时间差
            delta = time.time() - t1
            t1 = time.time()

            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 1. 图像预处理
                # 将整张图(frame)直接送入预处理，而不是先裁剪手部区域
                processed_frame = Demo.preprocess(frame, transform)

                # 2. 模型推理
                with torch.no_grad():  # 禁用梯度计算
                    # 模型输入需要增加一个 Batch 维度: [C, H, W] -> [1, C, H, W]
                    output = classifier([processed_frame])

                # 3. 获取预测结果
                # output["labels"] 通常包含每个类别的 Logits 或概率
                # argmax(dim=1) 找出概率最大的那个类别的索引
                label = output["labels"].argmax(dim=1)

                # 4. 可视化结果
                # 直接在屏幕固定位置 (10, 100) 显示当前画面的分类结果
                cv2.putText(
                    frame,
                    targets[int(label)],     # 将数字索引转为对应的文字（如 "Stop"）
                    (10, 100),               # 文本坐标
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,                       # 字体大小
                    (0, 0, 255),             # 颜色 (红色)
                    thickness=3
                )
                # 计算并显示 FPS
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, (255, 0, 255), 2)
                cnt += 1

                # 显示图像窗口
                cv2.imshow("Frame", frame)

                # 按 'q' 键退出
                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                # 如果无法读取帧（摄像头断开或视频结束），释放资源
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Demo full frame classification...")
    # 必须参数：配置文件路径
    parser.add_argument("-p", "--path_to_config", required=False, type=str, default="hagrid_v2/configs/se_resnet18.yaml", help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    # 1. 获取参数
    args = parse_arguments()
    # 2. 加载 YAML 配置
    conf = OmegaConf.load(args.path_to_config)
    # 3. 构建模型
    # 注意：这里的配置文件应该指向一个分类模型（如 ResNet），而不是检测模型
    model = build_model(conf)
    # 4. 构建预处理变换
    transform = Demo.get_transform_for_inf(conf.test_transforms)
    # 5. 加载模型权重
    if conf.model.checkpoint is not None:
        # map_location="cpu" 保证在无 GPU 的机器上也能运行
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(snapshot["MODEL_STATE"])
    # 6. 设置为评估模式
    model.eval()
    # 7. 运行演示
    if model is not None:
        Demo.run(model, transform)
