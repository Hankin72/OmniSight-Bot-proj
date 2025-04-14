import cv2
import os
import time
import threading
import numpy as np
# from gevent import pywsgi
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

from FaceDatabase import DEFAULT_FILENAME_DB, save_face_database, get_image_paths, load_face_database
from insightface.app import FaceAnalysis
from FaceUtils import LOCAL_MODELS_PATH, draw_colored_landmarks
from servo.PCA9685 import PCA9685
from models.myRknnFaceAnalysis import MyRknnFaceAnalysis
from face_loader import get_face_model

app = Flask(__name__)
CORS(app)

CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
TEMPLATES_DIR = os.path.join(CURR_PATH, "templates")
UPLOAD_FOLDER = os.path.join(CURR_PATH, 'collected_faces')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_MODULES=['detection', 'recognition', 'landmark_2d_106']
DRAW_LANDMARKS = False

# 模型加载
face_model = get_face_model(use_rknn=True, allowed_modules=['detection', 'recognition'], root='./models')

# 初始数据库
FACE_DB = load_face_database(DEFAULT_FILENAME_DB)

# 摄像头索引
USB_CAM_INDEX = 0

# 全局状态
detect_streaming = False
track_streaming = False

# 舵机控制初始化
pwm = PCA9685(0x40)
pwm.setPWMFreq(50)

PAN_CHANNEL, TILT_CHANNEL = 1, 0 # 水平方向舵机, # 垂直方向舵机
pan_angle, tilt_angle = 90, 70  # 水平方向舵机初始化角度, # 垂直方向舵机初始化角度


# 误差死区，防止舵机微小抖动
DEAD_ZONE = 40

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

def angle_to_pulse(angle):
    return int(600 + (angle / 180.0) * (2400 - 600))

def reset_servo_position():
    global pan_angle, tilt_angle
    pan_angle, tilt_angle = 90, 70
    pwm.setServoPulse(PAN_CHANNEL, angle_to_pulse(pan_angle))
    pwm.setServoPulse(TILT_CHANNEL, angle_to_pulse(tilt_angle))
    
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

    pwm.setServoPulse(PAN_CHANNEL, angle_to_pulse(pan_angle))
    pwm.setServoPulse(TILT_CHANNEL, angle_to_pulse(tilt_angle))
    

# 主页
@app.route("/")
def index():
    return render_template("index.html")


# 人脸采集页面
@app.route('/collect', methods=['POST', 'GET'])
def collect():
    return render_template('collect.html')


# 接收上传的人脸照片和姓名并按姓名创建文件夹
@app.route('/collect_upload', methods=['POST'])
def collect_upload():
    name = request.form.get('name')
    file = request.files.get('photo')

    if not name or not file:
        return jsonify({'status': 'error', 'message': 'Missing name or file'}), 400

    if not name.isalnum() and "_" not in name:
        return jsonify({'status': 'error', 'message': 'Invalid name format'}), 400
        
    user_folder = os.path.join(UPLOAD_FOLDER, name)    
    os.makedirs(user_folder, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{name}_{timestamp}.jpg"
    save_path = os.path.join(user_folder, filename)
    file.save(save_path)

    # 将该用户照片目录中的图片生成 embedding 并更新数据库
    image_paths = get_image_paths(user_folder)
    save_face_database(image_paths=image_paths, person_name=name, filename=DEFAULT_FILENAME_DB)

    return jsonify({'status': 'success', 'message': f'Successfuly! Image saved as {filename}'})


# # 提供上传页面（供手机扫码跳转后使用）
# @app.route('/collect_upload_page', methods=['GET'])
# def collect_upload_page():
#     return render_template('collect_upload.html')

# 实时人脸检测与识别页面<br/>
@app.route('/detect')
def detect():
    return render_template('detect.html')


# 视频流接口
@app.route('/detect_feed')
def detect_feed():
    return Response(generate_detect_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detect_feed', methods=['POST'])
def stop_detect_feed():
    global detect_streaming
    detect_streaming = False
    print("stop detect called")
    return jsonify({'status': 'success', 'message': 'Detect Streaming stopped'})

# 人脸追踪页面
@app.route('/track')
def track():
    return render_template('track.html')

@app.route('/track_feed/<target_name>')
def track_feed(target_name):
    print("track_feed =====> ", target_name)
    return Response(generate_track_stream(target_name), mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/stop_track_feed', methods=['POST'])
def stop_track_feed():
    global track_streaming
    track_streaming = False
    print("stop tracking called")
    # reset_servo_position()
    time.sleep(0.8)  # Small delay to allow camera release
    return jsonify({'status': 'success', 'message': 'Track Streaming stopped'})


# 刷新人脸数据库接口
@app.route('/refresh_face_db', methods=['POST'])
def refresh_face_db():
    global FACE_DB
    print("refresh_face_db called")
    FACE_DB = load_face_database(DEFAULT_FILENAME_DB)
    return jsonify({'status': 'success', 'message': 'Face database refreshed successfully'})
    
@app.route('/get_known_names')
def get_known_names():
    return jsonify({'names': list(FACE_DB.keys())})

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

    
# # 实时检测流
def generate_detect_stream():
    global detect_streaming
    
    cap = cv2.VideoCapture(USB_CAM_INDEX)
    if not cap.isOpened(): raise IOError("Cannot open webcam")
    detect_streaming = True
    
    while detect_streaming:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        # frame = cv2.flip(frame, -1)
        
        faces = face_model.get(frame)

        # face_model.draw_on(frame, faces)
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            if DRAW_LANDMARKS:
                # 绘制关键点
                lmk = face.landmark_2d_106
                lmk = np.round(lmk).astype(int)
                draw_colored_landmarks(frame, lmk)

            embedding = face.embedding
            if embedding is not None:
                name, sim = recognize_face(embedding)
                color = (0, 0, 255) if name != "Unknown" else (0, 255, 0)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# 实时追踪流
def generate_track_stream(target_name):
    global track_streaming
    cap = cv2.VideoCapture(USB_CAM_INDEX)
    if not cap.isOpened(): raise IOError("Cannot open webcam")
    track_streaming = True
    while track_streaming:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
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
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.circle(frame, (fx, fy), 5, (255, 255, 0), -1)  # 标记人脸中心
                cv2.line(frame, (cx, cy), (fx, fy), (255, 255, 255), 2)  # 连线
                
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    

if __name__ == "__main__":
    reset_servo_position()
    
    PORT = 16880
    SERVER = "0.0.0.0"
    
    app.run(host=SERVER, port=PORT, debug=True)

    # server = pywsgi.WSGIServer((SERVER, PORT), app)
    #
    # #
    # # print(f"running server on localhost  : http://127.0.0.1:{PORT}")
    # # print(f"running server on            : http://172.18.199.220:{PORT}")
    # # print(f"running server on            : http://192.168.31.124:{PORT}")
    # #
    # server.serve_forever()
    # print(TEMPLATES_DIR)

