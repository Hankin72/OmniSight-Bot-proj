import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import onnxruntime as ort
from FaceUtils import *
from FaceDatabase import load_face_database, DEFAULT_FILENAME_DB
from models.myRknnFaceAnalysis import MyRknnFaceAnalysis
from face_loader import get_face_model

DEFAULT_FILENAME_DB = 'my_face_database_s.npy'
USE_DEFAULT_MODEL = "buffalo_s"
FACE_DATABASE = load_face_database(filename=DEFAULT_FILENAME_DB)
DRAW_LANDMARKS = False


class FaceRecognitionTest:
    def __init__(self,
                 model_name='buffalo_s',
                 allowed_modules=['detection', 'landmark_2d_106'],
                 providers=None,
                 ctx_id=0,
                 det_size=(640, 640),
                 camera_index=0,
                 flip=True):
        """
        初始化人脸检测器。
        参数：
        - model_name: 使用的模型名称，默认 'buffalo_l'，可选 'buffalo_s'（轻量级模型）
        - allowed_modules: 允许的模块列表，如 ['detection', 'landmark_2d_106']
        - providers: ONNX Runtime 的计算提供商列表
        - ctx_id: 计算设备 ID，-1 表示使用 CPU，0 表示使用第一个 GPU
        - det_size: 人脸检测的输入尺寸 (width, height)
        - camera_index: 摄像头设备索引，默认 0
        - flip: 是否水平翻转帧，默认 True
        """
        if PI_PROVIDERS:
            self.providers = PI_PROVIDERS
        else:
            self.providers = providers if providers else ort.get_available_providers()

        # print('Available providers:', self.providers)
        
        # 模型加载， 默认为RKNN
        self.app = get_face_model()

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("无法打开摄像头")

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, det_size[0])
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, det_size[1])

        self.flip = flip
        self.threshold = .45

    def process_frame(self, frame):
        """
        处理单帧图像，进行人脸检测和关键点绘制。
        """
        # 人脸检测
        faces = self.app.get(frame)

        # 绘制检测结果
        for face in faces:
            bbox = face.bbox.astype(int)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_GREEN, 1)

            if DRAW_LANDMARKS:
                # 绘制关键点
                lmk = face.landmark_2d_106
                lmk = np.round(lmk).astype(int)

                draw_colored_landmarks(frame, lmk)

            embedding = face.embedding
            if embedding is None:
                print("Error: 无法提取特征向量")
                continue

            name, pre = self.recognize_face(embedding)
            # pre = "{:.2f}".format(pre)
            # temp_label = f"{name} {pre}"
            # print(temp_label)
            if name == "Unknown":
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)
            else:
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
        return frame

    @staticmethod
    def cosine_similarity(a, b):
        """
        计算两个向量之间的余弦相似度
        参数:
        - a: 向量 a
        - b: 向量 b
        返回:
        - similarity: 余弦相似度 (介于 -1 和 1 之间)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def recognize_face(self, embedding):
        if embedding is None:
            print("Error: 提取的 embedding 为 None")
            return 'Unknown'

        best_match = None
        highest_similarity = -1
        for name, embeddings in FACE_DATABASE.items():
            for known_embedding in embeddings:
                if known_embedding is None:
                    print(f"Warning: {name} 的 embedding 为 None，跳过")
                    continue
                similarity = self.cosine_similarity(embedding, known_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = name
        if highest_similarity >= self.threshold:
            return best_match, highest_similarity
        else:
            return 'Unknown', highest_similarity

    def run(self):
        """
        运行实时人脸检测。
        """
        # if image_path:
        #     img = cv2.imread(image_path)
        #     img = self.process_frame(img)
        #     cv2.imshow('Real-time Face Detection', img)
        #     show_image(img)
        #     # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     #     return
        # else:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if self.flip:
                frame = cv2.flip(frame, 1)

            frame = self.process_frame(frame)

            # self.out.write(frame)

            cv2.imshow('Real-time Face Detection', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.release()

    def release(self):
        self.cap.release()
        # self.video_writer.release()  # 释放视频写入器
        cv2.destroyAllWindows()

    def __del__(self):
        self.release()


if __name__ == '__main__':
    detector = FaceRecognitionTest(
        model_name=USE_DEFAULT_MODEL,
        allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
        ctx_id=-1,  # 设置为 -1 使用 CPU
        det_size=(640, 480),
        camera_index=0,
        flip=True
    )
    detector.run()
