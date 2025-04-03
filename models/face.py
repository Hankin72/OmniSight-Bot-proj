# face.py

import numpy as np

class Face:
    def __init__(self, bbox=None, kps=None, det_score=None):
        """
        bbox: (x1, y1, x2, y2)
        kps: (5, 2)
        det_score: float
        """
        self.bbox = np.array(bbox, dtype=np.float32) if bbox is not None else None
        self.kps = np.array(kps, dtype=np.float32) if kps is not None else None
        self.det_score = det_score

        self.embedding = None                # 人脸特征（face feature）
        self.landmark_2d_106 = None          # 106关键点坐标 shape=(106, 2 or 3)
        self.age = None                      # 年龄（可扩展）
        self.gender = None                   # 性别（可扩展）

    @property
    def sex(self):
        if self.gender is None:
            return "unknown"
        return "female" if self.gender == 0 else "male"

    def __str__(self):
        return f"<Face bbox={self.bbox} score={self.det_score:.2f if self.det_score is not None else 'N/A'}>"
