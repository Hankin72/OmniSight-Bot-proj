from face_detect_rknn import face_detect
from rknn.api import RKNN
import cv2
import numpy as np

def load_rknn_model(model_path, target='rk3588'):
    rknn = RKNN()
    print(f"--> Loading RKNN model: {model_path}")
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError("Failed to load RKNN model")
    ret = rknn.init_runtime(target=target)
    if ret != 0:
        raise RuntimeError("Failed to init RKNN runtime")
    print("RKNN model loaded and initialized.")
    return rknn

def face_align_cv2(img, keypoints, image_size=(112, 112), method="similar"):
    target_pts = np.float32([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ])
    src_pts = np.float32(keypoints)
    if method == "affine":
        M = cv2.getAffineTransform(src_pts[:3], target_pts[:3])
    else:
        M, _ = cv2.estimateAffinePartial2D(src_pts, target_pts, method=cv2.LMEDS)
    aligned_face = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    if len(aligned_face.shape) == 2:
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2BGR)
    return aligned_face

def face_feature(aligned_faces, feature_rknn):
    features = []
    for face in aligned_faces:
        face = face.astype(np.float32)
        face = (face - 127.5) / 128
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        net_outs = feature_rknn.inference(inputs=[face])
        features.append(net_outs[0])
    return features

if __name__ == "__main__":
    det_model_path = "./model/buffalo_sc/det_500m.rknn"
    feature_model_path = "./model/buffalo_sc/w600k_mbf.rknn"
    img_path = "./dataset/2.jpg"

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

    for i, face in enumerate(aligned_faces):
        cv2.imshow(f"Aligned Face {i+1}", face)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
