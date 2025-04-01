# face_landmark_rknn.py
# 用于使用 2d106det.rknn 模型进行 106 点人脸关键点检测

import cv2
import numpy as np

# 1) 从 face_detect_rknn 导入
#    - face_detect(): 用 det_500m.rknn 做人脸检测
#    - load_rknn_model(): 读取 .rknn 并 init_runtime
from face_detect_rknn import face_detect, load_rknn_model

# 2) 从 face_align.py 导入你已有的对齐函数/仿射变换函数
from face_align import transform, trans_points
from face_feature_rknn import face_align_cv2, face_feature



def get_face_landmark_106(img, bbox, input_size=(192, 192), lmk_num=106, landmark_rknn=None):
    """
    使用 2d106det.rknn 对单个人脸进行 106 点关键点检测.
    可选步骤
    """
    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
    rotate = 0
    scale = input_size[0] / (max(w, h) * 1.5)
    
    print("param:", img.shape, bbox, center, input_size, scale, rotate)
    
    # 仿射变换 => aimg
    aimg, M = transform(img, center, input_size[0], scale, rotate)

    # blob
    input_size = tuple(aimg.shape[0:2][::-1])
    
    blob = cv2.dnn.blobFromImage(aimg, scalefactor=1.0, size=input_size, mean=(0,0,0), swapRB=True)
    blob = blob.transpose(0, 2, 3, 1)  # => NHWC
    
    out_list = landmark_rknn.inference(inputs=[blob], data_format='nhwc')
    
    # 输出转换为关键点坐标 
    pred = out_list[0]
    
    print("rknn_landmark_outs", len(pred), pred.shape)

    # reshape => (106,2) or (106,3)
    if pred.shape[0] >= 3000:
        pred = pred.reshape((-1, 3))
    else:
        pred = pred.reshape((-1, 2))

    if lmk_num < pred.shape[0]:
        pred = pred[lmk_num * -1:,:]

    pred[:,0:2] += 1
    pred[:,0:2] *= input_size[0]//2
    if pred.shape[1]==3:
        pred[:,2] *= input_size[0]//2

    # inverse transform
    IM = cv2.invertAffineTransform(M)
    pred = trans_points(pred, IM)
    return pred

def main():
    # =============== 配置区 ==================
    # 1) 人脸检测模型
    DET_RKNN_PATH = "/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/det_500m.rknn"
    # 2) 人脸特征模型 (w600k_mbf.rknn)
    FEATURE_RKNN_PATH = "/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/w600k_mbf.rknn"
    # 3) (可选) 106 点关键点模型
    LANDMARK_106_RKNN_PATH = "/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/2d106det.rknn"
    # 测试图片
    IMG_PATH = "/home/orangepi/Documents/new_face_algorithm_proj/dataset/01_IMG_1029.JPG"

    # =============== 加载图 & 模型 ===============
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"无法读取图片: {IMG_PATH}")
        return

    # 加载检测 & 特征 & (可选) 106-landmark
    det_rknn = load_rknn_model(DET_RKNN_PATH, target='rk3588')
    feature_rknn = load_rknn_model(FEATURE_RKNN_PATH, target='rk3588')
    landmark_106_rknn = load_rknn_model(LANDMARK_106_RKNN_PATH, target='rk3588')

    # ============== 1. 人脸检测 ================
    # 这里 face_detect 会返回 bboxes=[N,5], keypoints=[N,5,2]
    # 其中 bboxes[i] = (x1,y1,x2,y2,score)
    #       keypoints[i] = 5个关键点 (眼睛, 鼻尖, 嘴角)
    bboxes, keypoints_5 = face_detect(
        img, input_size=(640,640), max_num=0, det_rknn=det_rknn, use_kps=True
    )

    # ============== 2. (可选) 106关键点检测 ================
    # 如果你只需要 5 个关键点来对齐，可以跳过此步
    # 这里以 for i in range(len(bboxes)) 演示
    all_landmarks_106 = []
    for i, bbox in enumerate(bboxes):
        print(i, "----->", bbox[:4])
        # bbox[:4] => [x1,y1,x2,y2]
        # shape => (4,)
        lmk_106 = get_face_landmark_106(img, bbox[:4], landmark_rknn=landmark_106_rknn)
        all_landmarks_106.append(lmk_106)

    # ============== 3. 人脸对齐 & 特征提取 ================
    # 对齐使用 5 个关键点 (RetinaFace 自带)
    aligned_faces = []
    for kp5 in keypoints_5:
        aligned = face_align_cv2(img, kp5, (112,112))
        aligned_faces.append(aligned)

    features = face_feature(aligned_faces, feature_rknn)

    # ============== 4. 可视化 ================
    # 4.1 在原图上画 (x1,y1,x2,y2) + score
    for i, (bbox, kp5) in enumerate(zip(bboxes, keypoints_5)):
        x1,y1,x2,y2,score = bbox
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0),2)
        cv2.putText(img, f"Face {i+1} s={score:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # 在图上画 5 点
        for (kx, ky) in kp5:
            cv2.circle(img, (int(kx), int(ky)), 2, (0,255,255), -1)

    # 4.2 如果使用 106 点检测，画在原图上
    for lmk_106 in all_landmarks_106:
        for (lx, ly) in np.round(lmk_106).astype(np.int32):
            cv2.circle(img, (lx,ly), 1, (0,0,255), -1)

    
    # 4.3 在终端打印特征的前五维
    for i, feat in enumerate(features):
        print(f"[Face {i+1}] embedding (first 5 dims): {feat[0][:5]}")
    

    # 4.4 显示结果
    cv2.imshow("Face Detection + 106 Landmark + Feature", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ============== 5. 释放资源 ================
    det_rknn.release()
    feature_rknn.release()
    landmark_106_rknn.release()

if __name__ == "__main__":
    main()
