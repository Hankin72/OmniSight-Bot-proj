import gi
import cv2
import numpy as np
import threading
import time
import queue
from gi.repository import Gst

from YOLOV8_POSE.yolov8_pose import Yolov8PoseRKNN
from face_loader import get_face_model
from FaceDatabase import load_face_database
from FaceUtils import *

# 初始化 GStreamer
Gst.init(None)

# 配置摄像头
CAMERA_DEVICE = "/dev/video0"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 使用 tee 插件分流视频帧，后续可拓展到多模型同步处理
pipeline_str = (
    f"v4l2src device={CAMERA_DEVICE} ! "
    f"video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT} ! "
    "tee name=t t. ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=face_sink max-buffers=1 drop=true "
    "t. ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=pose_sink max-buffers=1 drop=true "
)

pipeline = Gst.parse_launch(pipeline_str)

face_sink = pipeline.get_by_name("face_sink")

pose_sink = pipeline.get_by_name("pose_sink")

for sink in [face_sink, pose_sink]:
    sink.set_property("emit-signals", False)
    sink.set_property("sync", False)

# 模型加载
pose_model = Yolov8PoseRKNN("./YOLOV8_POSE/yolov8n-pose_int8.rknn", target="rk3588", verbose=True)
face_model = get_face_model(int_8=True, allowed_modules = ['detection', 'recognition'])
FACE_DATABASE = load_face_database("my_face_database_s.npy")

# 队列和结果
face_frame_queue = queue.Queue(maxsize=1)
pose_frame_queue = queue.Queue(maxsize=1)
face_result = {}
pose_result = {}
result_lock = threading.Lock()

THRESHOLD = 0.45

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    return dot / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(embedding):
    best_match, highest = None, -1
    for name, embs in FACE_DATABASE.items():
        for e in embs:
            if e is None:
                continue
            sim = cosine_similarity(embedding, e)
            if sim > highest:
                highest, best_match = sim, name
    return (best_match, highest) if highest >= THRESHOLD else ("Unknown", highest)

def draw_faces(faces, frame):
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), COLOR_GREEN, 1)
        name, sim = recognize_face(face.embedding)
        label = f"{name}" if name != "Unknown" else name
        color = COLOR_RED if name != "Unknown" else COLOR_GREEN
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def appsink_to_frame(sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return None
    buf = sample.get_buffer()
    caps = sample.get_caps()
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        return None
    try:
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
        return cv2.flip(frame, 1)
    finally:
        buf.unmap(mapinfo)

# 人脸识别线程
def face_worker():
    while True:
        frame = appsink_to_frame(face_sink)
        if frame is None:
            continue
        input_frame = frame.copy()
        start = time.time()
        faces = face_model.get(input_frame)
        elapsed = (time.time() - start) * 1000
        print(f"[FaceWorker] Found {len(faces)} faces, 人脸识别耗时: {elapsed:.3f} ms")
        with result_lock:
            face_result["faces"] = faces
            face_result["frame"] = input_frame

# 姿态识别线程
def pose_worker():
    while True:
        frame = appsink_to_frame(pose_sink)
        if frame is None:
            continue
        start = time.time()
        boxes, drawed = pose_model.infer(frame, need_draw=True)
        elapsed = (time.time() - start) * 1000
        print(f"[PoseWorker] Found {len(boxes)} person, 姿态识别耗时: {elapsed:.3f} ms")
        with result_lock:
            pose_result["image"] = drawed
            pose_result["timestamp"] = time.time()

# 显示线程
def display_loop():
    pTime = time.time()
    
    last_ts = 0 
    frame_out = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    while True:
        # frame_out = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        with result_lock:
            if "image" in pose_result:
                frame_out = pose_result["image"]
                current_ts = pose_result.get("timestamp", 0)
            else:
                current_ts = last_ts
                
            if "faces" in face_result:
                draw_faces(face_result["faces"], frame_out)
                
        if current_ts != last_ts and current_ts != 0:
            fps = 1.0 / (current_ts - last_ts)
            last_ts = current_ts
        else:
            fps = 0
        
        if fps:
            # print(f'FPS: {int(fps)}')
            cv2.putText(frame_out, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
            
        cv2.imshow("OmniSight-Camera", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🚀 启动 GStreamer + 多线程分流推理系统")
    pipeline.set_state(Gst.State.PLAYING)
    threading.Thread(target=face_worker, daemon=True).start()
    threading.Thread(target=pose_worker, daemon=True).start()
    display_loop()
