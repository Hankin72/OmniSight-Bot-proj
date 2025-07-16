import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
import time

import sys
from pathlib import Path
from YOLOV8_POSE.yolov8_pose import Yolov8PoseRKNN
from face_loader import get_face_model
from FaceDatabase import load_face_database, DEFAULT_FILENAME_DB
from FaceUtils import *


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

THRESHOLD = .45

def recognize_face(embedding):
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
            similarity = cosine_similarity(embedding, known_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = name
    if highest_similarity >= THRESHOLD:
        return best_match, highest_similarity
    else:
        return 'Unknown', highest_similarity
        

def draw_faces(faces):
            # 绘制检测结果
    for face in faces: 
        bbox = face.bbox.astype(int)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_GREEN, 1)

        embedding = face.embedding
        if embedding is None:
            print("Error: 无法提取特征向量")
            continue

        name, pre = recognize_face(embedding)
        if name == "Unknown":
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)
        else:
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)

# 初始化 GStreamer, 必须调一次
Gst.init(None)

pipeline_str = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,width=640,height=480 ! "
    "videoconvert ! video/x-raw,format=RGB ! "
    "appsink name=sink max-buffers=1 drop=true"
)


pipeline = Gst.parse_launch(pipeline_str)

appsink = pipeline.get_by_name("sink")
appsink.set_property("emit-signals", False)
appsink.set_property("sync", False)

pipeline.set_state(Gst.State.PLAYING)



model_path = "/home/orangepi/Documents/OmniSight-Bot-proj/YOLOV8_POSE/yolov8n-pose_int8.rknn"
# model_path_hybrid = "./yolov8n-pose_hybrid_int8.rknn"
pose_model = Yolov8PoseRKNN(model_path=model_path, target="rk3588", verbose=True)


face_model = get_face_model(int_8=True)

DEFAULT_FILENAME_DB = 'my_face_database_s.npy'
USE_DEFAULT_MODEL = "buffalo_s"
FACE_DATABASE = load_face_database(filename=DEFAULT_FILENAME_DB)


print("🚀 摄像头已启动，按 'q' 退出")
# --- 新增: FPS 计算变量 ---
pTime = 0
cTime = 0
while True:
    sample = appsink.emit("pull-sample")
    if sample is None:
        continue

    buf = sample.get_buffer()
    caps = sample.get_caps()
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    try:
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))
        

        frame =  cv2.flip(frame, 1)
        
        start_time = time.time()
        
        faces = face_model.get(frame)
        draw_faces(faces=faces)
        # frame = face_model.draw_on(frame, faces)
        # frame = face_model.draw_on_landmark106(frame, faces)
    
        # boxes, drawed_image = pose_model.infer(frame, need_draw=True)
        
        inference_time = (time.time() - start_time) * 1000

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime =  cTime
        
        # print(f", FPS: {fps:.2f}")
        
        print(f"Found 0 person, {len(faces)} faces, Inference time: {inference_time:.4f}ms, FPS: {fps:.3f}")
        cv2.putText(frame, f'FPS: {int(fps)}', (40, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    finally:
        buf.unmap(mapinfo)

pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
