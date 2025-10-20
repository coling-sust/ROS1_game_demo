#!/usr/bin/env python3
import os
from ultralytics import YOLO

def export_to_tensorrt(model_path, output_dir="trt_engine", half=False, dynamic=False):
    """
    将 YOLO 模型导出为 TensorRT 格式
    :param model_path: 训练好的 .pt 模型路径
    :param output_dir: 输出引擎文件保存目录
    :param half: 是否使用 FP16 精度
    :param dynamic: 是否使用动态 batch 尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 正在加载模型: {model_path}")

    model = YOLO(model_path)

    print("⚙️ 正在导出为 TensorRT 引擎...")
    model.export(
        format="engine",       # 导出 TensorRT 格式
        half=half,             # 是否使用 FP16
        dynamic=dynamic,       # 是否使用动态 batch
        device=0,              # GPU 设备
        imgsz=640,             # 图像尺寸
        project=output_dir,    # 输出目录
        name="yolo_trt_model"  # 输出文件夹名
    )

    print("✅ TensorRT 引擎导出完成！")
    print(f"📁 输出路径: {os.path.join(output_dir, 'yolo_trt_model')}")

if __name__ == "__main__":
    # 示例：你的YOLO模型路径
    model_path = "/home/xyxy/game_ws/src/game_demo/scripts/yolo_weight/yolo11s_game.pt"

    # 是否使用 FP16（TensorRT 支持半精度可加速）
    export_to_tensorrt(model_path, half=True, dynamic=False)
