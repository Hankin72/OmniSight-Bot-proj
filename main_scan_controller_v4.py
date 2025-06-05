# main_scan_controller_v4.py

import threading
import time
import cv2
import numpy as np
from FaceUtils import *
from pcb.Pcb_Car import Pcb_Car
from FaceDatabase import load_face_database
from insightface.app import FaceAnalysis

car = Pcb_Car()

DEFAULT_FILENAME_DB_PI5 = 'my_face_database_s_pi5.npy'
USE_DEFAULT_MODEL = "buffalo_s"
FACE_DB = load_face_database(filename=DEFAULT_FILENAME_DB_PI5)
# 模型加载
PI_PROVIDERS = ['CPUExecutionProvider']

face_model = FaceAnalysis(name=USE_DEFAULT_MODEL, allowed_modules=['detection', 'recognition'], root="./models", providers=PI_PROVIDERS)
face_model.prepare(ctx_id=-1, det_size=(640, 480))

# 摄像头索引
USB_CAM_INDEX = 0

# 舵机编号
S1 = 1  # 垂直方向
S2 = 2  # 水平方向

PAN_CHANNEL, TILT_CHANNEL = S2, S1 # 水平方向舵机, # 垂直方向舵机
HORIZONTAL_SERVO_CENTER, VERTICAL_SERVO_INIT = 90, 40  # 水平方向舵机初始化角度, # 垂直方向舵机初始化角度
pan_angle, tilt_angle = HORIZONTAL_SERVO_CENTER, VERTICAL_SERVO_INIT  # 水平方向舵机初始化角度, # 垂直方向舵机初始化角度

# 旋转配置
SPIN_SPEED = 40
SPIN_DURATION = 8  # 秒，旋转一圈时间

# 跳帧检测间隔
DETECT_INTERVAL = 3

# DEAD_ZONE 死区范围
DEAD_ZONE = 40

# PID控制参数
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


class RotateFaceScanner:
    def __init__(self):

        self.current_pan_angle = HORIZONTAL_SERVO_CENTER
        self.current_tilt_angle = VERTICAL_SERVO_INIT
        
        self.running = False

    def reset_servo_position(self):
        self.set_vertical_angle()
        self.set_horizontal_angle()
        pid_pan.prev_error  = 0
        pid_pan.integral    = 0
        pid_tilt.prev_error = 0 
        pid_tilt.integral   = 0
        time.sleep(0.05) 
        
        
    def set_vertical_angle(self, angle=VERTICAL_SERVO_INIT):
        car.Ctrl_Servo(S1, angle)
        self.current_tilt_angle = angle

    def set_horizontal_angle(self, angle=HORIZONTAL_SERVO_CENTER):
        car.Ctrl_Servo(S2, angle)
        self.current_pan_angle = angle
        
    @staticmethod
    def compare_cosine_similarity(curr_emb, detected_emb):
        threshold = 0.45
        
        sim = np.dot(curr_emb, detected_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(detected_emb))
        if sim > threshold:
            return True
                
        return False
    
    def update_servo_position(self, face_x, face_y, center_x, center_y):
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
    

    def rotate_and_scan(self, direction='left'):
        global SPIN_DURATION, pan_angle, tilt_angle
        
        SPIN_DURATION = 8
        print("调整舵机初始位置...")
        self.reset_servo_position()
        
        print("开始旋转并检测人脸...")
        
        if direction == 'left':
            car.Car_Spin_Left(SPIN_SPEED, SPIN_SPEED)
        elif direction == 'right':
            car.Car_Spin_Right(SPIN_SPEED, SPIN_SPEED)
        else:
            print("无效方向")
            return

        self.running = True
        
        cap = cv2.VideoCapture(USB_CAM_INDEX)
        if not cap.isOpened(): raise IOError("Cannot open webcam")
        
        pTime = 0
        frame_count = 0
        every_interval = 3
        
        start_time = time.time()
        
        detected_face = None
        detected_face_embedding =  None
        
        while self.running:
            d_time = time.time() - start_time
            if d_time > SPIN_DURATION:
                if not detected_face:
                    print("旋转超时，未检测到人脸")
                    self.running = False
                else:
                    print("旋转超时，检测到人脸, 完成追踪 。。。 ")
                    self.running = False
                car.Car_Stop()
            
            time.sleep(0.01)
            print(" spining time 0.01 ---> ", d_time)
            
                                
            ret, frame = cap.read()
            
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            frame_count += 1
            
            frame_height, frame_width = frame.shape[:2]
            cx, cy = frame_width // 2, frame_height // 2
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # 标记画面中心
            
            if frame_count % every_interval == 0:
                faces = face_model.get(frame)
                
                if faces:
                    if detected_face_embedding is None:
                        face = faces[0]
                        detected_face_embedding = face.embedding
                        
                        detected_face = face
                        
                        car.Car_Stop()
                        
                        # time.sleep(0.05) 
                
                        SPIN_DURATION += 3
                        print("旋转检测到人脸, PID追踪 + 3s")
                    else:
                        for face in faces:
                            if self.compare_cosine_similarity(face.embedding, detected_face_embedding):
                                # car.Car_Stop()
                                
                                bbox = face.bbox.astype(int)
                                fx = int((bbox[0] + bbox[2]) / 2)  # 人脸中心点（X）
                                fy = int((bbox[1] + bbox[3]) / 2)   # 人脸中心点（Y）
                                
                                self.update_servo_position(fx, fy, cx, cy)
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

                                cv2.circle(frame, (fx, fy), 5, (255, 255, 0), -1)  # 标记人脸中心
                                cv2.line(frame, (cx, cy), (fx, fy), (255, 255, 255), 2)  # 连线
                            
    
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(frame, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.imshow('Real-time Face Detection', frame)
            

            # if cv2.waitKey(1) == ord('q'):
            #     break


        car.Car_Stop()
        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print(" spining car stop ---> ")


    def track_face_loop(self, face_scanner):
        while True:
            ret, frame = face_scanner.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            faces = face_scanner.app.get(frame)
            if not faces:
                print("人脸丢失，结束追踪")
                break

            face = faces[0]
            self.center_face_pid(face, frame)

            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

                
    # def track_face_loop(self):
    #     while True:
    #         ret, frame = self.cap.read()
    #         if not ret: break
            
    #         frame = cv2.flip(frame, 1)
            
    #         faces = self.face_scanner.app.get(frame)
            
    #         frame_height, frame_width = frame.shape[:2]
    #         frame_center_x = frame_width // 2  #
    #         frame_center_y = frame_height // 2  #
    #         cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 255, 255), -1)  # 标记画面中心
            
    #         if not faces:
    #             print("人脸丢失， 结束追踪")
            
    #         face = faces[0]
            
    #         bbox = face.bbox.astype(int)
            
    #         face_center_x = int((bbox[0] + bbox[2]) / 2)
    #         face_center_y = int((bbox[1] + bbox[3]) / 2)
            
    #         self.center_face_pid(frame_center_x, frame_center_y, face_center_x, face_center_y)
            
    #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            
    #         cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 255, 0), -1)  # 标记人脸中心
            
    #         cv2.line(frame, (frame_center_x, frame_center_y), (face_center_x, face_center_y), (255, 255, 255), 2)  # 连线
            
    #         cv2.imshow("Face Tracking", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
            
             
            
    def center_face_pid(self, face, frame):
        bbox = face.bbox.astype(int)
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 255, 255), -1)  # 标记画面中心

        face_center_x = int((bbox[0] + bbox[2]) / 2)
        face_center_y = int((bbox[1] + bbox[3]) / 2)

        offset_x = face_center_x - frame_center_x
        offset_y = face_center_y - frame_center_y

        if abs(offset_x) < DEAD_ZONE and abs(offset_y) < DEAD_ZONE:
            return

        adjust_pan = pid_pan.update(offset_x)
        adjust_tilt = pid_tilt.update(offset_y)

        self.current_pan_angle += adjust_pan
        self.current_tilt_angle -= adjust_tilt

        self.current_pan_angle = max(0, min(180, self.current_pan_angle))
        self.current_tilt_angle = max(0, min(180, self.current_tilt_angle))

        car.Ctrl_Servo(S2, self.current_pan_angle)
        car.Ctrl_Servo(S1, self.current_tilt_angle)

        cv2.putText(frame, f"Pan: {self.current_pan_angle:.1f} Tilt: {self.current_tilt_angle:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)


# 模拟信号触发
if __name__ == '__main__':
    scanner = RotateFaceScanner()
    try:
        while True:
            cmd = input("输入 l 或 r 触发传感器信号 (q退出): ").strip()
            if cmd == 'l':
                scanner.rotate_and_scan(direction='left')
            elif cmd == 'r':
                scanner.rotate_and_scan(direction='right')
            elif cmd == 'q':
                break
            else:
                print("无效指令")
    finally:
        car.Car_Stop()