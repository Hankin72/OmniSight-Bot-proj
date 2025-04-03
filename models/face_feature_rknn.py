# face_feature_rknn.py
import cv2
import numpy as np

from .face_detect_rknn import face_detect
from .face_detect_rknn import load_rknn_model


def face_align_cv2(img, keypoints, image_size=(112, 112), method="similar"):
    """
    使用 OpenCV 进行人脸对齐
    :param img: 原始图像
    :param keypoints: 5 个人脸关键点 [[x1, y1], [x2, y2], ..., [x5, y5]]
    :param image_size: 对齐后的人脸尺寸
    :param method: "affine" 或 "similar" 变换方法
    :return: 112x112 对齐后的人脸图像
    """
    # 目标关键点（MobileFaceNet 标准 112x112 对齐点）
    target_pts = np.float32([
        [38.2946, 51.6963],  # 左眼中心
        [73.5318, 51.5014],  # 右眼中心
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ])
    # 转换输入关键点为np.array
    
    src_pts = np.float32(keypoints) 
    if method == "affine":
        M = cv2.getAffineTransform(src_pts[:3], target_pts[:3])
    else:
        M, _ = cv2.estimateAffinePartial2D(src_pts, target_pts, method=cv2.LMEDS)
        
    # 进行仿射变换，生成112x112的对齐人脸
    aligned_face = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    
    # 确保输出图图像格式正确
    if len(aligned_face.shape) == 2:
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2BGR)
        
    return aligned_face

def face_feature(aligned_faces, feature_rknn):
    """
    提取对齐后的人脸特征
    例如 w600k_mbf.rknn
    """
    features = []
    for i, face in enumerate(aligned_faces):
        # shape(3,112,112) -> shape(1,3,112,112)
        face = face.astype(np.float32)
        # face = (face - 127.5) / 128
        # face = np.transpose(face, (2, 0, 1))
        
        face = np.expand_dims(face, axis=0)
        # net_outs = feature_rknn.inference(inputs=[face], data_format='nchw')
        
        net_outs = feature_rknn.inference(inputs=[face], data_format='nhwc')
        
        # print(f"feature-{i}: {net_outs}")
        features.append(net_outs[0])
    return features


if __name__ == "__main__":
    det_model_path = "/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/det_500m.rknn"
    
    feature_model_path = "/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/w600k_mbf.rknn"
    
    img_path = "/home/orangepi/Documents/face_algorithm_proj/dataset/01_IMG_1029.JPG"

    img = cv2.imread(img_path)

    det_rknn = load_rknn_model(det_model_path, target='rk3588')
    feature_rknn = load_rknn_model(feature_model_path, target='rk3588')

    bboxes, keypoints = face_detect(
        img, input_size=(640, 640), max_num=0, det_rknn=det_rknn, use_kps=True
    )

    aligned_faces = [face_align_cv2(img, kp) for kp in keypoints]
    features = face_feature(aligned_faces, feature_rknn)


    det_rknn.release()
    feature_rknn.release()

    for i, (bbox, feature) in enumerate(zip(bboxes, features)):
        x1, y1, x2, y2, score = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Face {i}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        print(f"Face {i+1} embedding (first 5 dims): {feature[0][:5]}")
        
        # # 若有关键点
        # if keypoints is not None:
        #     kps = keypoints[i]  # (5,2)
        #     for (kx, ky) in kps:
        #         cv2.circle(img, (int(kx), int(ky)), 2, (0,0,255), -1)

    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
