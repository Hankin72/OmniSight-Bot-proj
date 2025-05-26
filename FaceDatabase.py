import os.path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from FaceUtils import *
import json
from models.myRknnFaceAnalysis import MyRknnFaceAnalysis
from face_loader import get_face_model


DEFAULT_FILENAME_DB = 'my_face_database_s.npy'

# DEFAULT_FILENAME_DB = 'my_face_database_sc.npy'
# USE_DEFAULT_MODEL="buffalo_sc"
USE_DEFAULT_MODEL = "buffalo_s"
RKNN_PLATFORM=True


class FaceDatabase:
    def __init__(self):
        """
        初始化人脸数据库，数据库以字典形式存储姓名和对应的特征向量。
        """
        self.known_faces_db = {}

    def add_face(self, name, embedding):
        if name in self.known_faces_db:
            self.known_faces_db[name].append(embedding)  # 如果该人名已经存在，则添加新特征
        else:
            self.known_faces_db[name] = [embedding]  # 否则新建一个列表
        print(f">>>>> Add face to db: {name}")

    def save_to_file(self, filename=DEFAULT_FILENAME_DB):
        """
        将人脸数据库保存到文件
        - filename: 保存文件的路径 (默认为 {DEFAULT_FILENAME_DB})
        """
        np.save(filename, self.known_faces_db)
        print(f">>>> face db saved to file: {filename}")

    def load_from_file(self, filename=DEFAULT_FILENAME_DB):
        """
        加载人脸数据库
        - filename: 文件路径
        """
        self.known_faces_db = np.load(filename, allow_pickle=True).item()
        print(f"已从文件 {filename} 加载人脸数据库")
        return self.known_faces_db


class FaceDetector:
    def __init__(self, model_name=USE_DEFAULT_MODEL):
        
        self.app = get_face_model(int_8=True)

    def detect_and_extract(self, image_path):
        """
        从照片中检测人脸并提取特征向量
        参数:
        - image_path: 照片路径
        返回:
        - faces: 检测到的人脸对象列表 (包含 bounding box, embedding 等)
        """
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        if len(faces) == 0:
            print("没有检测到人脸")
        return faces


def get_image_paths(directory):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def save_face_database(image_paths=[], person_name="", filename=DEFAULT_FILENAME_DB):
    """
    将多个图片的 embedding 存入人脸数据库。
    参数:
    - image_paths: 图片路径列表
    - person_name: 对应的用户姓名
    """
    face_db = FaceDatabase()

    if os.path.exists(filename):
        face_db.load_from_file()
    else:
        print(f"{filename} not exists, create a face-db")

    detector = FaceDetector()

    for image_path in image_paths:
        print(">>>>> Processing image_path: " + image_path)
        faces = detector.detect_and_extract(image_path)

        # faces 可能为 None 或空列表，要先判断
        if not faces:
            continue

        # 新增过滤条件：若检测到的人脸数 > 1，则打印提示并跳过保存
        if len(faces) > 1:
            print(f"检测到超过 1 个人脸，跳过保存 => {image_path}")
            continue

        # 仅当检测到 1 张人脸的情况下，才进行保存
        if len(faces) == 1:
            embedding = faces[0].embedding
            face_db.add_face(person_name, embedding)

    # 保存
    face_db.save_to_file()


def load_face_database(filename=""):
    if os.path.exists(filename):
        face_db = FaceDatabase()
        return face_db.load_from_file(filename=filename)
    return None


if __name__ == '__main__':
    # image_paths = get_image_paths("./sample/haojin")

    # save_face_database(image_paths=image_paths, person_name="haojin", filename=DEFAULT_FILENAME_DB)

    # 1. 读取 JSON 文件
    with open('faces_paths.json', 'r', encoding='utf-8') as f:
        face_paths_dict = json.load(f)

    # 2. 遍历字典，逐个执行人脸数据库保存
    for person_name, image_folder_path in face_paths_dict.items():
        image_paths = get_image_paths(image_folder_path)  # 使用之前的 get_image_paths 函数
        print(f"当前处理姓名: {person_name}, 对应图片目录: {image_folder_path}")
        save_face_database(image_paths=image_paths, person_name=person_name, filename=DEFAULT_FILENAME_DB)
