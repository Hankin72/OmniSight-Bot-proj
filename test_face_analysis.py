
import cv2
import os
import time
import threading
import numpy as np
from FaceDatabase import DEFAULT_FILENAME_DB, save_face_database, get_image_paths, load_face_database
from insightface.app import FaceAnalysis
from FaceUtils import LOCAL_MODELS_PATH, draw_colored_landmarks
from pcb.Pcb_Car import Pcb_Car

DEFAULT_FILENAME_DB_PI5 = 'my_face_database_s_pi5.npy'
USE_DEFAULT_MODEL = "buffalo_s"
FACE_DB = load_face_database(filename=DEFAULT_FILENAME_DB_PI5)
# 模型加载
PI_PROVIDERS = ['CPUExecutionProvider']

face_model = FaceAnalysis(name=USE_DEFAULT_MODEL, allowed_modules=['detection', 'recognition'], root="./models", providers=PI_PROVIDERS)
face_model.prepare(ctx_id=-1, det_size=(640, 640))

# 摄像头索引
USB_CAM_INDEX = 0

# 全局状态
detect_streaming = False
track_streaming = False


PAN_CHANNEL, TILT_CHANNEL = 2, 1 # 水平方向舵机, # 垂直方向舵机
pan_angle, tilt_angle = 90, 50  # 水平方向舵机初始化角度, # 垂直方向舵机初始化角度


# 误差死区，防止舵机微小抖动
DEAD_ZONE = 40

car = Pcb_Car()


# PID 控制器参数
dt = 0.1
class PID:
    def __init__(self, kp, ki, kd, output_limit=5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # 限制输出最大变化幅度
        output = max(-self.output_limit, min(self.output_limit, output))
        return output


pid_pan = PID(0.015, 0.01, 0.0008)
pid_tilt = PID(0.025, 0.01, 0.0083)


# 识别函数
def recognize_face(embedding):
    global FACE_DB
    best_match = "Unknown"
    highest_sim = -1
    threshold = 0.45

    for name, embeddings in FACE_DB.items():
        for known_emb in embeddings:
            sim = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
            if sim > highest_sim:
                highest_sim = sim
                best_match = name

    if highest_sim < threshold:
        best_match = "Unknown"

    return best_match, highest_sim

def reset_servo_position():
    global pan_angle, tilt_angle
    pan_angle, tilt_angle = 90, 50
    car.Ctrl_Servo(PAN_CHANNEL, pan_angle)
    car.Ctrl_Servo(TILT_CHANNEL, tilt_angle)
    
    pid_pan.prev_error  = 0
    pid_pan.integral    = 0
    pid_tilt.prev_error = 0 
    pid_tilt.integral   = 0
    time.sleep(1.0)  # 舵机回中后，等待1秒再开始 PID 控制

def update_servo_position(face_x, face_y, center_x, center_y):
    global pan_angle, tilt_angle
    dx = face_x - center_x
    dy = face_y - center_y

    if abs(dx) < DEAD_ZONE: dx = 0
    if abs(dy) < DEAD_ZONE: dy = 0

    pan_angle += pid_pan.update(dx)
    tilt_angle += pid_tilt.update(dy)

    pan_angle = max(0, min(180, pan_angle))
    tilt_angle = max(0, min(180, tilt_angle))

    car.Ctrl_Servo(PAN_CHANNEL, pan_angle)
    car.Ctrl_Servo(TILT_CHANNEL, tilt_angle)
    
    
def generate_track_stream(target_name="hankin"):
    global track_streaming
    cap = cv2.VideoCapture(USB_CAM_INDEX)
    if not cap.isOpened(): raise IOError("Cannot open webcam")
    track_streaming = True
    
    pTime = 0

    frame_count = 0
    every_interval = 3
    while track_streaming:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        frame_count += 1
        
        if frame_count % every_interval == 0:

            faces = face_model.get(frame)

            frame_height, frame_width = frame.shape[:2]
            cx, cy = frame_width // 2, frame_height // 2
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # 标记画面中心
            
            for face in faces:
                name, sim = recognize_face(face.embedding)
                bbox = face.bbox.astype(int)
                if name == target_name:
                    fx = int((bbox[0] + bbox[2]) / 2)  # 人脸中心点（X）
                    fy = int((bbox[1] + bbox[3]) / 2)   # 人脸中心点（Y）
                    
                    update_servo_position(fx, fy, cx, cy)
                    
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                    cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    cv2.circle(frame, (fx, fy), 5, (255, 255, 0), -1)  # 标记人脸中心
                    cv2.line(frame, (cx, cy), (fx, fy), (255, 255, 255), 2)  # 连线
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow('Real-time Face Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
                
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    generate_track_stream()