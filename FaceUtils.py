import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FACE_RANGE = range(0, 33)
LIP_RANGE = range(52, 72)
NOSE_RANGE = range(72, 87)
EYE_LEFT_RANGE = range(33, 43)
EYE_RIGHT_RANGE = range(87, 97)
EYEBROW_LEFT_RANGE = range(97, 106)
EYEBROW_RIGHT_RANGE = range(43, 52)

FACE_OUTLINE_COLOR = (255, 0, 0)  # 脸部轮廓：蓝色 (255, 0, 0)
LIP_COLOR = (0, 165, 255)  # 嘴唇：橙色 (0, 165, 255)
EYE_COLOR = (0, 255, 0)  # 眼睛：绿色 (0, 255, 0)
EYEBROW_COLOR = (0, 255, 255)  # 眉毛：黄色 (0, 255, 255)
NOSE_COLOR = (235, 206, 135)  # 鼻子

COLOR_RED = (0, 0, 255)  # 绘图颜色，绿色
COLOR_GREEN = (0, 255, 0)

BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
MODELS=['buffalo_s', 'buffalo_l']
USE_DEFAULT_MODEL = MODELS[0]
LOCAL_MODELS_PATH = "./models"
ALLOW_MODULES=['detection', 'recognition', 'landmark_2d_106']
PI_PROVIDERS = ['CPUExecutionProvider']

import onnxruntime as ort
PROVIDERS = []
PROVIDERS = PROVIDERS if PROVIDERS else ort.get_available_providers()

def draw_on(img, face):
    import cv2
    import numpy as np
    dimg = img.copy()
    if face is not None:
        box = face.bbox.astype(np.int64)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(np.int64)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                           2)
        if face.gender is not None and face.age is not None:
            cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 255, 0), 1)
    return dimg


def draw_colored_landmarks(frame, lmk):
    """
    绘制不同颜色的关键点。
    参数：
    - frame: 输入的图像帧
    - lmk: 关键点坐标数组
    """
    # 绘制脸部轮廓关键点
    for i in FACE_RANGE:
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, FACE_OUTLINE_COLOR, -1)

    # 绘制嘴唇关键点
    for i in LIP_RANGE:
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, LIP_COLOR, -1)

    # 绘制眼睛关键点
    for i in EYE_LEFT_RANGE:
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, EYE_COLOR, -1)
    for i in EYE_RIGHT_RANGE:
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, EYE_COLOR, -1)

    # 绘制眉毛关键点
    for i in EYEBROW_LEFT_RANGE:  # 左眉毛
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, EYEBROW_COLOR, -1)

    for i in EYEBROW_RIGHT_RANGE:  # 左眉毛
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, EYEBROW_COLOR, -1)

    # 绘制鼻子关键点
    for i in NOSE_RANGE:
        p = tuple(lmk[i])
        cv2.circle(frame, p, 1, NOSE_COLOR, -1)


def show_image(image, figsize=(13, 13)):
    # 创建指定大小的画布
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print(EYE_LEFT_RANGE)
    print(EYE_RIGHT_RANGE)
    print()
