# face_loader.py

import os
from typing import List, Optional, Tuple

from models.myRknnFaceAnalysis import MyRknnFaceAnalysis
from insightface.app import FaceAnalysis  # ONNX版本

# 默认平台（RKNN=True 表示用 OrangePi 或 Rockchip 平台）
RKNN_PLATFORM = True
DEFAULT_MODEL_NAME = 'buffalo_s'


def get_face_model(
    use_rknn: bool = RKNN_PLATFORM,
    model_name: str = DEFAULT_MODEL_NAME,
    allowed_modules: Optional[List[str]] = None,
    root: str = './models',
    det_size: Tuple[int, int] = (640, 640),
    ctx_id: int = -1
):
    """
    初始化人脸分析模型接口

    Args:
        use_rknn (bool): 是否使用 RKNN 模型（否则使用 ONNX）。
        model_name (str): 模型名称，例如 'buffalo_s'。
        allowed_modules (list): ['detection', 'recognition', 'landmark_2d_106']。
        root (str): 模型根目录路径。
        det_size (tuple): 检测输入图像尺寸。
        ctx_id (int): ONNX 模式下的 GPU/CPU 配置，-1 代表 CPU。

    Returns:
        实例化好的模型对象，可调用 .get(img), .draw_on(img, faces)
    """
    if allowed_modules is None:
        allowed_modules = ['detection', 'recognition', 'landmark_2d_106']
    
    if use_rknn:
        model = MyRknnFaceAnalysis(name=model_name, allowed_modules=allowed_modules, root=root)
        model.prepare(det_size=det_size)
    else:
        model = FaceAnalysis(name=model_name, allowed_modules=allowed_modules, root=root)
        model.prepare(ctx_id=ctx_id, det_size=det_size)
    
    return model
