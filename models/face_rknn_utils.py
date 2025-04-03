import matplotlib.pyplot as plt
import cv2


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

COLOR_RED = (0, 0, 255)  # 绘图颜色，红色
COLOR_GREEN = (0, 255, 0) # 绘图颜色，绿色


def showimage(image, figsize=(13, 13)):
    # 创建指定大小的画布
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

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