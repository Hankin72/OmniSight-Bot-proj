from flask import Flask, render_template, request, jsonify, send_from_directory
from gevent import pywsgi
from flask_cors import CORS
import cv2
import threading
import os
import time
import numpy as np
from utils import kill_process_on_port


app = Flask(__name__)
CORS(app)

CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
TEMPLATES_DIR = os.path.join(CURR_PATH, "templates")


# 主页，
@app.route("/", methods=['POST','GET'])
def index():
    return send_from_directory(TEMPLATES_DIR, "index.html")
    # return render_template("index.html")

# 人脸采集
@app.route('/collect', methods=['POST', 'GET'])
def collect():
    if request.method == "POST":
        return render_template('collect.html')

    return render_template('collect.html')
    

# 实时人脸检测与识别
@app.route('/detect')
def detect():
    return render_template('detect.html')


# 处理人脸检测请求
@app.route('/detect_feed')
def detect_feed():
    return ""


# 人脸追踪
@app.route('/track', methods=['POST','GET'])
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