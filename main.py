from flask import Flask, render_template, request, jsonify, send_from_directory
from gevent import pywsgi
from flask_cors import CORS
import cv2
import threading
import os
import time
import numpy as np
from utils import kill_process_on_port
from FaceDatabase import DEFAULT_FILENAME_DB, save_face_database, get_image_paths

app = Flask(__name__)
CORS(app)

CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
TEMPLATES_DIR = os.path.join(CURR_PATH, "templates")

UPLOAD_FOLDER = os.path.join(CURR_PATH, 'collected_faces')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 主页
@app.route("/", methods=['POST', 'GET'])
def index():
    return send_from_directory(TEMPLATES_DIR, "index.html")

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

    # 校验名称避免非法字符
    if not name.isalnum() and "_" not in name:
        return jsonify({'status': 'error', 'message': 'Invalid name format; use letters, numbers or underscores only.'}), 400

    user_folder = os.path.join(UPLOAD_FOLDER, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    timestamp = int(time.time())
    filename = f"{name}_{timestamp}.jpg"
    save_path = os.path.join(user_folder, filename)
    file.save(save_path)
    
    # 将该用户照片目录中的图片生成 embedding 并更新数据库
    image_paths = get_image_paths(user_folder)
    save_face_database(image_paths=image_paths, person_name=name, filename=DEFAULT_FILENAME_DB)

    return jsonify({'status': 'success', 'message': f'Image saved in {user_folder} as {filename}'})

# # 提供上传页面（供手机扫码跳转后使用）
# @app.route('/collect_upload_page', methods=['GET'])
# def collect_upload_page():
#     return render_template('collect_upload.html')

# 实时人脸检测与识别页面<br/>
@app.route('/detect')
def detect():
    return render_template('detect.html')

# 人脸追踪页面
@app.route('/track', methods=['POST', 'GET'])
def tracking():
    return render_template('track.html')
    

if __name__ == "__main__":
    
    PORT = 16880
    SERVER = "0.0.0.0"
    kill_process_on_port(PORT)
    
    server = pywsgi.WSGIServer((SERVER, PORT), app)
    
    print("running server on localhost  : http://127.0.0.1:16880")
    print("running server on            : http://172.18.199.220:16880")
    print("running server on            : http://192.168.31.124:16880")
    
    
    server.serve_forever()
    # print(TEMPLATES_DIR)
    
    # app.run(debug=True)