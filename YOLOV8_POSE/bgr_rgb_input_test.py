#!/usr/bin/env python3
# bgr_vs_rgb_diff.py
import cv2, time, numpy as np, math
from yolov8_pose import Yolov8PoseRKNN

# ---------- 参数 ----------
MODEL_PATH       = './yolov8n-pose_int8.rknn'
CAM_ID           = 0
NUM_FRAMES_TEST  = 200        # 设为 0 表示无限循环直到 Ctrl+C
IOU_THRESH_PAIR  = 0.5
SHOW_FIRST_DIFF  = True       # 首次出现差异时弹窗可视化

# ---------- IoU 工具 ----------
def iou(box1, box2):
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1.xmax-box1.xmin) * (box1.ymax-box1.ymin)
    area2 = (box2.xmax-box2.xmin) * (box2.ymax-box2.ymin)
    return inter / (area1+area2-inter+1e-6)

# ---------- 初始化 ----------
pose = Yolov8PoseRKNN(MODEL_PATH, target='rk3588', verbose=False)

video_path = "/home/orangepi/Documents/pose_tracking_on_rknn/videos/ori.mp4"
# cap = cv2.VideoCapture(filename=video_path)

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

sum_iou, paired, diff_count, frame_idx = 0, 0, 0, 0

# ---------- 主循环 ----------
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("摄像头丢帧, 退出"); break
    frame_idx += 1
    frame_rgb  = frame_bgr[..., ::-1]            # 通道翻转

    # 1. BGR 推理
    boxes_bgr, _ = pose.infer(frame_bgr, need_draw=False)

    # 2. RGB 推理
    boxes_rgb, _ = pose.infer(frame_rgb, need_draw=False)

    # 3. 统计比较
    if len(boxes_bgr) != len(boxes_rgb):
        diff_count += 1
        if SHOW_FIRST_DIFF:
            print(f'[Frame {frame_idx}] count diff: BGR={len(boxes_bgr)}, RGB={len(boxes_rgb)}')
            SHOW_FIRST_DIFF = False

    # 用 IoU 做一一配对（贪心，简单够用）
    used = set()
    for b in boxes_bgr:
        best_iou, best_j = 0, -1
        for j, r in enumerate(boxes_rgb):
            if j in used: continue
            v = iou(b, r)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou > IOU_THRESH_PAIR:
            sum_iou += best_iou
            paired += 1
            used.add(best_j)
        else:
            diff_count += 1
            if SHOW_FIRST_DIFF:
                print(f'[Frame {frame_idx}] unmatched box, IoU={best_iou:.2f}')
                SHOW_FIRST_DIFF = False

    if NUM_FRAMES_TEST and frame_idx >= NUM_FRAMES_TEST:
        break

print('\n==== 统计结果 ====')
print(f'测试帧数           : {frame_idx}')
print(f'检测框总数差异帧数 : {diff_count}')
if paired:
    print(f'平均匹配 IoU       : {sum_iou/paired:.3f} (共 {paired} 对)')
else:
    print('未找到可配对框 (阈值 IoU>0.5)')

cap.release()
pose.release()
