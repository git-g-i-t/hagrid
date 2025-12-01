""""
实时手势识别演示（Demo）脚本

功能：
1. 加载配置文件和预训练模型。
2. 打开摄像头捕获视频流。
3. 对每一帧使用预训练的 PyTorch 模型检测手部位置并识别手势
4. (可选) 使用 MediaPipe 绘制手部骨骼。
5. 将模型输出的检测框还原回原图坐标并绘制。
"""

#基础库
import argparse   #命令行参数
import logging    #日志
import time       #计算FPS
from typing import Optional, Tuple


import albumentations as A                      # 图像增强与预处理库
import cv2                                      # OpenCV, 用于图像处理和显示
import mediapipe as mp                          # 用于绘制手部骨骼关键点（Landmarks）
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2   # 将图片转为 PyTorch Tensor 的工具
from omegaconf import DictConfig, OmegaConf     # 配置管理库，用于读取 yaml 配置文件
from torch import Tensor

# MediaPipe 的绘图工具初始化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 导入项目自定义模块
# targets: 手势类别名称列表，映射类别索引到字符串（如 0 -> "call", 1 -> "like"）
from constants import targets
# build_model: 根据配置构建模型架构的工厂函数
from custom_utils.utils import build_model

# build_model: 根据配置构建模型架构的工厂函数
logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

# 定义可视化时的颜色 (绿色) 和字体
COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        模型输入的预处理函数
        Parameters
        ----------
        img: np.ndarray
            摄像头的原始图像帧
        transform :
            albumentations 的变换管道（通常包含 Resize, Padding 等）

        Returns
        -------
        processed_image: Tensor
            归一化并转换维度的张量，准备送入模型
        (width, height): Tuple
            原始图像的宽和高，用于后续坐标还原
        """
        height, width = img.shape[0], img.shape[1]                # 记录原始尺寸
        transformed_image = transform(image=img)                  # 应用预处理（Resize, Pad等）
        processed_image = transformed_image["image"] / 255.0      # 归一化：将像素值从 [0, 255] 缩放到 [0, 1]
        return processed_image, (width, height)

    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        """
        Create list of transforms from config
        从配置文件构建推理用的变换管道

        Parameters
        ----------
        transform_config: DictConfig
            config with test transforms
            配置文件中的 test_transforms 部分
        """
        # 动态读取配置中的变换名称（如 LongestMaxSize）并实例化
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        # 最后添加转为 Tensor 的步骤
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(
        detector, transform, conf: DictConfig, num_hands: int = 2, threshold: float = 0.5, landmarks: bool = False
    ) -> None:
        """
        Run detection model and draw bounding boxes on frame
        运行检测模型并在帧上绘制边界框

        Parameters
        ----------
        detector :
            加载好的 PyTorch 检测模型
        transform :
            albumentation transforms 预处理管道
        conf: DictConfig
            完整配置对象，用于获取输入尺寸等参数
            config with test transforms
        num_hands: int
            最多检测几只手
        threshold : float
             置信度阈值，低于此分数的检测框会被忽略
        landmarks : bool
            是否开启 MediaPipe 绘制骨骼关键点(landmarks)
        """

        # 如果开启了 landmarks 参数，初始化 MediaPipe Hands 模型
        if landmarks:
            hands = mp.solutions.hands.Hands(
                model_complexity=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8
            )

        # 打开默认摄像头 (ID 0)
        cap = cv2.VideoCapture(0)

        # 初始化时间变量用于计算 FPS
        t1 = cnt = 0

        # 开始视频流处理循环
        while cap.isOpened():
            # 计算帧间隔时间
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                # 1. 图像预处理
                processed_image, size = Demo.preprocess(frame, transform)

                # 2. 模型推理
                with torch.no_grad():  # 禁用梯度计算，节省显存并加速
                    # detector输入是一个 list，取 [0] 获得第一张图的结果
                    output = detector([processed_image])[0]

                # 获取前 num_hands 个预测结果
                boxes = output["boxes"][:num_hands]     # 边界框坐标 [x1, y1, x2, y2]
                scores = output["scores"][:num_hands]   # 置信度分数
                labels = output["labels"][:num_hands]   # 类别索引

                # 3. (可选) MediaPipe 骨骼绘制逻辑
                if landmarks:
                    # MediaPipe 需要 RGB 格式，OpenCV 默认是 BGR
                    results = hands.process(frame[:, :, ::-1])
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                                mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1),
                            )

                # 4. 遍历检测到的每一只手，绘制边界框和标签
                for i in range(min(num_hands, len(boxes))):
                    # 只有置信度大于阈值才显示
                    if scores[i] > threshold:
                        # --- 坐标还原逻辑开始 ---

                        # 模型输入通常经过了 Letterbox 处理（保持长宽比缩放 + 填充黑边）
                        # 我们需要将模型输出的坐标映射回摄像头原始画面的坐标系
                        width, height = size      # 原始宽高
                        scale = max(width, height) / conf.LongestMaxSize.max_size   # 计算缩放比例（基于最长边）

                        # 计算模型输入图像中填充（Padding）的大小
                        # 原始图缩放后的宽 = width // scale
                        # 填充的宽 = (目标宽 - 缩放后宽) / 2
                        padding_w = abs(conf.PadIfNeeded.min_width - width // scale) // 2
                        padding_h = abs(conf.PadIfNeeded.min_height - height // scale) // 2

                        # 反算坐标：
                        # 1. 减去填充量 (boxes[i] - padding)
                        # 2. 乘回缩放比例 (* scale)
                        x1 = int((boxes[i][0] - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)
                        # --- 坐标还原逻辑结束 ---

                        # 绘制矩形框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)

                        # 绘制类别标签     targets[int(labels[i])] 将数字索引转换为文字（如 "call"）
                        cv2.putText(
                            frame,
                            targets[int(labels[i])],
                            (x1, y1 - 10),     # 文字位置在框上方
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),       # 红色文字
                            thickness=3,
                        )

                # 计算并显示 FPS
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cnt += 1

                # 显示最终画面
                cv2.imshow("Frame", frame)

                # 按 'q' 键退出循环
                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                # 如果读取不到帧，释放资源
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    """
       命令行参数解析
    """
    parser = argparse.ArgumentParser(description="Demo detection...")

    # 必须参数：配置文件的路径
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    # 可选参数：是否开启 MediaPipe 关键点显示
    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    # 1. 获取命令行参数
    args = parse_arguments()
    # 2. 加载配置文件 (YAML)
    conf = OmegaConf.load(args.path_to_config)
    # 3. 构建模型结构
    model = build_model(conf)
    # 4. 获取预处理变换
    transform = Demo.get_transform_for_inf(conf.test_transforms)
    # 5. 加载模型权重 (Checkpoint)
    if conf.model.checkpoint is not None:
        # map_location="cpu" 确保即使没有 GPU 也能加载权重
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        # 加载状态字典
        model.load_state_dict(snapshot["MODEL_STATE"])

    # 6. 设置为评估模式 (关闭 Dropout, BatchNorm 统计固定)
    model.eval()

    # 7. 启动演示主循环
    if model is not None:
        Demo.run(model, transform, conf.test_transforms, num_hands=100, threshold=0.8, landmarks=args.landmarks)
