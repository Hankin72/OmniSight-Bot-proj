from PCA9685 import PCA9685
import cv2
import numpy as np
import onnxruntime as ort
import time
from insightface.app import FaceAnalysis
from FaceUtils import *
from FaceDatabase import load_face_database
import os
import pickle
import threading
from PIDController import PIDController
from ServoController import ServoController
import queue 

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

MODELS_PATH = "./faceRecognitionTest/local_model"
PI_PROVIDERS = ['CPUExecutionProvider']

FILENAME_DB = 'my_face_database_sc.npy'    # 本地人脸数据库
FACE_DATABASE = load_face_database(filename=FILENAME_DB)
APP_MODEL_PATH = 'face_analysis_model_sc.npy'  # # 人脸分析模型存储路径

# 舵机通道
PAN_CHANNEL = 1  # 水平舵机通道
TILT_CHANNEL = 0  # 垂直舵机通道

# PID 参数
Kp, Ki, Kd = 0.1, 0.005, 0.02  # 适用于舵机平滑控制
# PID 相关参数（可根据实际硬件做微调）
panP, panI, panD = 0.015, 0.01, 0.0008
tiltP, tiltI, tiltD = 0.025, 0.01, 0.0083

# 舵机角度限制
MAX_ANGLE = 180
MIN_ANGLE = 0
RESET_ANGLE = 90  # 默认位置（初始化）

# 误差死区，防止舵机微小抖动
DEAD_ZONE = 30

# **人脸检测间隔**
DETECTION_INTERVAL = 3  # 每 5 帧检测一次

# **无人脸超时复位**
FACE_TIMEOUT = 5  # 超过 5 秒无目标，舵机回到初始位置

MAX_STEP = 10  # 限制单次舵机调整最多 3°


# 线程安全的队列（主线程显示图像）
frame_queue = queue.Queue()


class ServoFaceTracker:
    """人脸追踪 + 舵机控制"""
    
    def __init__(self,
                 model_name='buffalo_sc',
                 allowed_modules=['detection', 'landmark_2d_106'],
                 ctx_id=0,
                 det_size=(640, 480),
                 threshold = .45):
        # **1. 初始化人脸识别模型**
        # 检查是否已有保存的 FaceAnalysis 模型
        if os.path.exists(APP_MODEL_PATH):
            print("Loading FaceAnalysis model from saved file...")
            with open(APP_MODEL_PATH, 'rb') as f:
                self.app = pickle.load(f)
        else:
            print("Initializing FaceAnalysis model and saving for future use...")
            self.app = FaceAnalysis(name=model_name, allowed_modules=allowed_modules, providers=['CPUExecutionProvider'], root=MODELS_PATH)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            with open(APP_MODEL_PATH, 'wb') as f:
                pickle.dump(self.app, f)  # 保存对象到文件
        
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.threshold = threshold

             

         # **3. 初始化舵机控制**
        self.servo = ServoController()
        self.servo.reset_servo()
        self.current_pan = RESET_ANGLE
        self.current_tilt = RESET_ANGLE

        # **4. 初始化 PID 控制**
       # 设置一级舵机的PID参数
        self.pan_pid = PIDController(panP, panI, panD)

        # 设置二级舵机的PID参数
        self.tilt_pid = PIDController(tiltP, tiltI, tiltD)

        # self.pan_pid = PIDController(Kp, Ki, Kd)
        # self.tilt_pid = PIDController(Kp, Ki, Kd)
        
        self.running = True
        self.capture_thread = threading.Thread(target=self.run_capture)
        self.capture_thread.start()

        # **5. 额外参数**
        self.frame_count = 0  # 记录帧数
        self.last_detected_faces = []  # 缓存最近一次检测结果
        self.last_face_time = time.time()  # 记录上次检测到人脸的时间

    def smooth_move(self,current_angle, target_angle):
        SMOOTH_FACTOR = 0.2  # 平滑系数
        return current_angle + SMOOTH_FACTOR * (target_angle - current_angle)
        
    def update_servo(self, face_bbox, frame_size):
        """计算误差 + PID 计算 + 更新舵机"""
        frame_center_x = frame_size[1] / 2   # 画面中心点（X）
        frame_center_y = frame_size[0] / 2   # 画面中心点（Y）
        face_center_x = (face_bbox[0] + face_bbox[2]) / 2    # 人脸中心点（X）
        face_center_y = (face_bbox[1] + face_bbox[3]) / 2    # 人脸中心点（Y）

        # 计算偏差（误差）
        error_x = face_center_x - frame_center_x
        error_y = face_center_y - frame_center_y

        # 仅当偏差超过 DEAD_ZONE 时才调整舵机
        if abs(error_x) >= DEAD_ZONE:
            delta_pan = self.pan_pid.update(error_x)
            delta_pan = max(-MAX_STEP, min(MAX_STEP, delta_pan))  # 限制单次调整幅度
            
            self.current_pan = self.smooth_move(self.current_pan, self.current_pan + delta_pan)
            
            self.current_pan = max(MIN_ANGLE, min(MAX_ANGLE, self.current_pan))
            self.servo.set_servo_angle(PAN_CHANNEL, self.current_pan)
            time.sleep(0.02)

        if abs(error_y) >= DEAD_ZONE:
            delta_tilt = self.tilt_pid.update(error_y)
            delta_tilt = max(-MAX_STEP, min(MAX_STEP, delta_tilt))  # 限制单次调整幅度
            
            self.current_tilt = self.smooth_move(self.current_tilt, self.current_tilt + delta_tilt)
            
            self.current_tilt = max(MIN_ANGLE, min(MAX_ANGLE, self.current_tilt))
            self.servo.set_servo_angle(TILT_CHANNEL, self.current_tilt)
            time.sleep(0.02)

        # **舵机超出视野范围时，自动复位**
        if self.current_pan < 10 or self.current_pan > 170 or self.current_tilt < 10 or self.current_tilt > 170:
            print("舵机超出范围，复位")
            self.servo.reset_servo()
        
        time.sleep(0.1)

    def process_frame(self, frame):
        """人脸检测与识别"""
        (H, W) = frame.shape[:2]
        # 在画面中心点绘制一个小圆点
        centerX, centerY = W // 2, H // 2
        cv2.circle(frame, (centerX, centerY), 5, (0, 255, 255), -1)

        self.frame_count += 1
        start_time = time.time()  # 记录帧开始时间

        if self.frame_count % DETECTION_INTERVAL == 0:
            faces = self.app.get(frame)  # 每 DETECTION_INTERVAL 帧检测一次
            self.last_detected_faces = faces
        else:
            faces = self.last_detected_faces  # 复用上一次检测结果

        # faces = self.app.get(frame)
        target_face = None

        # 绘制检测结果
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_GREEN, 1)

            embedding = face.embedding
            if embedding is None:
                print("Error: 无法提取特征向量")
                continue

            name, similarity = self.recognize_face(embedding)
            if name == "Unknown":
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)
            else:
                target_face = bbox
                face_center_x = (bbox[0] + bbox[2]) // 2    # 人脸中心点（X）
                face_center_y = (bbox[1] + bbox[3]) // 2    # 人脸中心点（Y）
                cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

                cv2.line(frame, (centerX, centerY), (face_center_x, face_center_y), COLOR_RED, 2)
                
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
        if target_face is not None:
            self.update_servo(target_face, frame.shape)

        # **计算并显示 FPS**
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        # **将帧放入队列，交给主线程显示**
        frame_queue.put(frame)
        # return frame

    @staticmethod
    def cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def recognize_face(self, embedding):
        """人脸特征匹配"""
        best_match = "Unknown"
        highest_similarity = -1
        for name, embeddings in FACE_DATABASE.items():
            for known_embedding in embeddings:
                similarity = self.cosine_similarity(embedding, known_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = name
        if highest_similarity >= self.threshold:
            return best_match, highest_similarity
        else:
            return 'Unknown', highest_similarity

    def run_capture(self):
        while self.running:
            # 捕获当前帧并处理
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.process_frame(frame)
            
    def release(self):
        self.picam2.stop()
    
    def __del__(self):
        self.release()

if __name__ == '__main__':
    detector = ServoFaceTracker()
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow('Real-time Face Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
