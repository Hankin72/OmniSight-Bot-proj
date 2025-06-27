from yolov8_pose import Yolov8PoseRKNN
import cv2 
import time
import numpy as np 

model_path = "./yolov8n-pose_int8.rknn"
model_path_hybrid = "./yolov8n-pose_hybrid_int8.rknn"

pose_model = Yolov8PoseRKNN(model_path=model_path, target="rk3588", verbose=True)


video_path = "/home/orangepi/Documents/pose_tracking_on_rknn/videos/ori.mp4"
# cap = cv2.VideoCapture(filename=video_path)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 试 640 或 640×640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 很多 USB 摄像头没有 1:1 分辨率
cap.set(cv2.CAP_PROP_FPS, 30)


if not cap.isOpened():
    raise IOError("无法打开USB摄像头, 请检查video index")

# --- 新增: FPS 计算变量 ---
pTime = 0
cTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧。正在退出...")
        break
    
    frame = cv2.flip(frame, 1)
    
    start_time = time.time()
    
    boxes, drawed_image = pose_model.infer(frame, need_draw=True)
    
    # boxes_old = pose_model.infer(frame, need_draw=False, use_fast=False)[0]
    # boxes_new = det.infer(frame, need_draw=False, use_fast=True)[0]

    # assert len(boxes_old)==len(boxes_new)
    # for b1,b2 in zip(boxes_old, boxes_new):
    #     assert np.allclose(
    #         [b1.xmin,b1.ymin,b1.xmax,b1.ymax],
    #         [b2.xmin,b2.ymin,b2.xmax,b2.ymax], atol=1e-3)
    # print("两套实现坐标完全一致！")

    inference_time = (time.time() - start_time) * 1000
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    
    pTime =  cTime
    
    print(f"Found {len(boxes)} person, Inference time: {inference_time:.4f}ms, FPS: {fps:.2f}")
    # print(f" Inference time: {inference_time:.4f}ms, FPS: {fps:.2f}")
    
    cv2.putText(drawed_image, f'FPS: {int(fps)}', (40, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('usb CAM detect', drawed_image)

    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pose_model.release()

