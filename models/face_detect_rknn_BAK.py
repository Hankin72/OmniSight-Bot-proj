# face_detect_rknn.py

import cv2
import numpy as np
from rknn.api import RKNN
import matplotlib.pyplot as plt

from .face_align import distance2bbox, distance2kps, nms
from .face_rknn_utils import showimage

INPUT_SIZE = (640, 640)  # (width, height)
STRIDES = [8, 16, 32]    # 常见RetinaFace的3个stride
CONF_THRESH = 0.5
NMS_THRESH = 0.4
    

def forward(img,threshold,fmc=3, feat_stride_fpn=[8, 16, 32], num_anchors=2,use_kps=False, det_rknn=None,):
    
    center_cache = {}
    global kps_preds
    scores_list = []
    bboxes_list = []
    kpss_list = []
    input_size = tuple(img.shape[0:2][::-1])

    # 将图像转换为模型可接受的输入格式
    blob = cv2.dnn.blobFromImage(
        img,
        1.0,
        input_size,
        (0, 0, 0),
        swapRB=True,
    )
    
    blob = blob.transpose(0, 2, 3, 1)  # Change from 'nchw' to 'nhwc'

    net_outs = det_rknn.inference(inputs=[blob], data_format='nhwc')
    # net_outs = det_rknn.inference(inputs=[blob], data_format='nhwc')

    input_height = blob.shape[1]
    input_width = blob.shape[2]
    
    for idx, stride in enumerate(feat_stride_fpn):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        bbox_preds = bbox_preds * stride
        if use_kps:
            kps_preds = net_outs[idx + fmc * 2] * stride
        height = input_height // stride
        width = input_width // stride
        # K = height * width
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
            print("key in center_cache")
        else:
        #     print("height, width, stride", height, width, stride)

            # solution-1, c style:
            anchor_centers = np.zeros((height, width, 2), dtype=np.float32)
            for i in range(height):
                anchor_centers[i, :, 1] = i
            for i in range(width):
                anchor_centers[:, i, 0] = i

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= threshold)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
    return scores_list, bboxes_list, kpss_list


# 获取人脸检测框
def face_detect(img, input_size=None, max_num=0, metric="default", use_kps=False, det_rknn=None, threshold=0.5):
    assert input_size is not None or input_size is not None
    
    input_size = input_size if input_size is None else input_size

    im_ratio = float(img.shape[0]) / img.shape[1]  # 计算输入图像的宽高比
    
    model_ratio = float(input_size[1]) / input_size[0] # 计算模型期望的高宽比
    
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
        
    det_scale = float(new_height) / img.shape[0]
    
    resized_img = cv2.resize(img, (new_width, new_height))
    
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    
    det_img[:new_height, :new_width, :] = resized_img

    # 调用forward对人脸进行检测
    scores_list, bboxes_list, kpss_list = forward(
        det_img, threshold=threshold, use_kps=use_kps, det_rknn=det_rknn
    )

    scores = np.vstack(scores_list) # 所有特征层的分数堆叠
    
    scores_ravel = scores.ravel()  # 展平分数数组
    
    order = scores_ravel.argsort()[::-1]  # 按照置信度进行排序
    
    bboxes = np.vstack(bboxes_list) / det_scale   # 还原边界框到原图尺度
    
    if use_kps:
        kpss = np.vstack(kpss_list) / det_scale
        
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    
    pre_det = pre_det[order, :]
    
    keep = nms(pre_det)
    
    det = pre_det[keep, :]
    
    if use_kps:
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]
    else:
        kpss = None 
        
    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack(
            [
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0],
            ]
        )
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        # If the metric is "max", prioritize the bounding boxes with the largest area
        if metric == "max":
            values = area
        else:
            values = (
                area - offset_dist_squared * 2.0
            )  # some extra weight on the centering
        bindex = np.argsort(values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]
        det = det[bindex, :]
        if kpss is not None:
            kpss = kpss[bindex, :]
            
            
    return det, kpss  # 返回最终检测框和关键点


def load_rknn_model(model_path, target='rk3588', core_mask=RKNN.NPU_CORE_0_1_2):
    rknn = RKNN()
    print(f"--> Loading RKNN model: {model_path}")
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError("Failed to load RKNN model")
    
    ret = rknn.init_runtime(target=target, core_mask=core_mask, async_mode=True)

    if ret != 0:
        raise RuntimeError("Failed to init RKNN runtime")
    
    # print("RKNN model loaded and initialized.")
    return rknn


def detect_faces(rknn_model_path, img_path, input_size=(640, 640), max_num=0, use_kps=False,):
    # 1. 加载模型
    rknn = load_rknn_model(rknn_model_path)

    # 3. 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to load image {img_path}")
        rknn.release()
        return None, None

    # 4. 检测
    det, kpss = face_detect(
        image,
        input_size=input_size,
        det_rknn=rknn,
        max_num=max_num,
        use_kps=use_kps,
    )

    rknn.release()

    if det is None or det.shape[0] == 0:
        print("No faces detected!")
        return None, None

    print("Detectded face num:\n", len(det))
    print("Bounding boxes:\n", det)
    
    # print("Face keypoints:\n", kpss)

    # # 保存检测框到 txt 文件
    # np.savetxt(output_bbox_path, det, fmt="%.5f")
    # print(f"Bounding boxes saved to {output_bbox_path}")

    # 可视化 => 画框 & 画关键点
    for i, face in enumerate(det):
        x1, y1, x2, y2, score = face
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制置信度，显示在检测框上方
        label = f"{score:.2f}"
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        # 若有关键点
        if use_kps and kpss is not None:
            kps = kpss[i]  # (5,2)
            for (kx, ky) in kps:
                cv2.circle(image, (int(kx), int(ky)), 2, (0,0,255), -1)

    # cv2.imshow("Result image", image)
    showimage(image)
    # cv2.imwrite(output_image_path, image)
    # print(f"Result image saved to {output_image_path}")

    return det, kpss
    

def main():
    # 1. 路径配置
    # RKNN_MODEL_PATH = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/det_500m.rknn'
    RKNN_MODEL_PATH = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/det_500m_int8.rknn'
    IMG_PATH = '/home/orangepi/Documents/face_algorithm_proj/dataset2/human_images/challenge2020-03-31-145224_0004.jpg'
    USE_KPS = True           # 是否解析关键点
    
    bbox_output_path = "/home/orangepi/Documents/face_algorithm_proj/models/results.txt"
    
    detect_faces(
        rknn_model_path=RKNN_MODEL_PATH,
        img_path=IMG_PATH,
        use_kps=USE_KPS,
    )



if __name__ == "__main__":
    main()