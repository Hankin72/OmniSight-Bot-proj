import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2,math
from math import ceil

from rknn.api import RKNN

CLASSES = ['person']

nmsThresh = 0.4
objectThresh = 0.5

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)
kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

class DetectBox:
    """检测目标 & 关键点包装"""
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint
        
# ---------------------- 推理类 ----------------------
class Yolov8PoseRKNN:
    def __init__(self, model_path: str,
                target: str = 'rk3588',
                device_id: str | None = None,
                verbose: bool = False):
        """加载 RKNN 模型并初始化运行时"""
        self.model_path = model_path
        
        # self.rknn = RKNN(verbose=verbose)
        self.rknn = RKNN(verbose=verbose)
        
        assert self.rknn.load_rknn(model_path) == 0, f'Load {model_path} failed'
        
        # MASK = (RKNN.NPU_CORE_0 | RKNN.NPU_CORE_1 | RKNN.NPU_CORE_2)  # 即 0b111
        MASK = RKNN.NPU_CORE_0_1_2
        assert self.rknn.init_runtime(
            target=target,
            device_id=device_id,
            core_mask=MASK) == 0, 'Init runtime failed'
    
    def release(self):         # 用完记得释放
        self.rknn.release()
        
    # ---------- 内部工具 ----------
    @staticmethod
    def _letterbox_resize(image, size=(640, 640), bg_color=56):
        """
        letterbox_resize the image according to the specified size
        :param image: input image, which can be a NumPy array or file path
        :param size: target size (width, height)
        :param bg_color: background filling data 
        :return: processed image
        """    
        if isinstance(image, str):
            image = cv2.imread(image)
        
        target_width, target_height = size
        image_height, image_width, _ = image.shape
        
        # Calculate the adjusted image size
        aspect_ratio = min(target_width / image_width, target_height / image_height)
        new_width = int(image_width * aspect_ratio)
        new_height = int(image_height * aspect_ratio)
        
        # Use cv2.resize() for proportional scaling
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a new canvas and fill it
        result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
        return result_image, aspect_ratio, offset_x, offset_y


    @staticmethod
    def _pad_640x480_to_640x640(img_bgr, bg=56):
        """输入 640×480 BGR，顶底各 pad 80 → 640×640"""
        out = np.full((640, 640, 3), bg, np.uint8)
        out[80:560, :, :] = img_bgr
        # 这里返回 ratio=1.0、offset_x=0、offset_y=80，方便后续坐标回映射
        return out, 1.0, 0, 80



    @staticmethod
    def _iou(a, b):
        xmin = max(a.xmin, b.xmin) 
        ymin = max(a.ymin, b.ymin)
        xmax = min(a.xmax, b.xmax)
        ymax = min(a.ymax, b.ymax)
        
        innerWidth = max(0, xmax-xmin)
        innerHeight = max(0, ymax-ymin)
        innerArea = innerWidth*innerHeight
        
        area1 = (a.xmax-a.xmin)*(a.ymax-a.ymin)
        area2 = (b.xmax-b.xmin)*(b.ymax-b.ymin)
        union = area1 + area2 - innerArea

        return innerArea / union if union > 0 else 0
    
    @staticmethod
    def _nms(boxes):
        predBoxs = []
        boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
        for i, box in enumerate(boxes):
            if box.classId < 0:   # 被抑制
                continue
            predBoxs.append(box)
            for j in range(i+1, len(boxes)):
                if box.classId == boxes[j].classId and Yolov8PoseRKNN._iou(box, boxes[j]) > nmsThresh:
                    boxes[j].classId = -1
        return predBoxs

    @staticmethod
    def _sigmoid(x):  
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _softmax(x, axis=-1):
        # 将输入向量减去最大值以提高数值稳定性
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    
    
    def _process_fast(self, feat, all_kpts, idx_off, feat_w, feat_h, stride, scale_w=1., scale_h=1.):
        # feat: (1, 64+1, H*W) → (4,16,H,W)
        xywh = feat[:, :64, :].reshape(4, 16, feat_h, feat_w)
        prob = self._sigmoid(feat[:, 64:, :]).reshape(1, 1, feat_h, feat_w)

        keep = prob[0, 0] > objectThresh
        if not keep.any():
            return []

        # softmax + weighted-sum
        bins = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
        xywh = (self._softmax(xywh, 1) * bins).sum(1)      # (4, H, W)

        grid_x, grid_y = np.meshgrid(np.arange(feat_w), np.arange(feat_h))

        # ---------- 与原 C 版公式一致 ----------
        xmin = ((grid_x + 0.5) - xywh[0])[keep]
        ymin = ((grid_y + 0.5) - xywh[1])[keep]
        xmax = ((grid_x + 0.5) + xywh[2])[keep]
        ymax = ((grid_y + 0.5) + xywh[3])[keep]

        scores = prob[0, 0][keep]
        pts_idx = np.flatnonzero(keep) + idx_off
        kpts = all_kpts[..., pts_idx]        # shape: (51, n_det)

        boxes = []
        for i, s in enumerate(scores):
            boxes.append(
                DetectBox(
                    classId=0,
                    score=float(s),
                    xmin=xmin[i] * stride * scale_w,
                    ymin=ymin[i] * stride * scale_h,
                    xmax=xmax[i] * stride * scale_w,
                    ymax=ymax[i] * stride * scale_h,
                    keypoint=kpts[..., i],
                )
            )
        return boxes
    
    
    def _process(self, feat, all_kpts, idx_offset, feat_w, feat_h, stride, scale_w=1., scale_h=1.):
        """将某一层特征图解析为 DetectBox 列表"""
        
        xywh = feat[:, :64, :]
        conf = self._sigmoid(feat[:, 64:, :])
        boxes = []
        
        c = 0
        
        for h in range(feat_h):
            for w in range(feat_w):
                # for c in range(len(CLASSES)):
                score = conf[0, c, h*feat_w + w]
                
                if score < objectThresh:
                    continue
                
                xywh_ = xywh[0, :, h*feat_w + w].reshape(1, 4, 16, 1)
                data = np.arange(16).reshape(1,1,16,1)
                xywh_ = self._softmax(xywh_, 2)
                xywh_ = np.multiply(data, xywh_)
                xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                # 还原到特征图坐标
                xywh_temp=xywh_.copy()
                xywh_temp[0]=(w+0.5)-xywh_[0]
                xywh_temp[1]=(h+0.5)-xywh_[1]
                xywh_temp[2]=(w+0.5)+xywh_[2]
                xywh_temp[3]=(h+0.5)+xywh_[3]

                xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                xywh_=xywh_*stride
                

                xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                
                kpt=all_kpts[..., (h*feat_w) + w + idx_offset]
                kpt[..., 0:2] //= 1
                boxes.append(DetectBox(classId=c, score=score, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, keypoint=kpt)) 
                
        return boxes
    

    # ---------- 核心接口 ----------
    def infer(self, img_bgr, need_draw=True, use_fast=True):
        """对单张 BGR 格式图像推理  
        返回 (boxes, vis_img)。若 need_draw=False，第二个返回值为 None
        """
        
        letterbox_img, aspect_ratio, offset_x, offset_y = self._letterbox_resize(img_bgr, (640,640), 56)  # letterbox缩放
        # infer_img = letterbox_img[..., ::-1]  # BGR->RGB

        # letterbox_img, aspect_ratio, offset_x, offset_y = self._pad_640x480_to_640x640(img_bgr, 56)
        infer_img = letterbox_img  # BGR->RGB
        
        # Inference
        # print('--> Running model')
        results = self.rknn.inference(inputs=[infer_img])
        
        
        outputs = []
        keypoints = results[3]
        for x in results[:3]:
            index,stride=0,0
            if x.shape[2]==20:
                stride=32
                index=20*4*20*4+20*2*20*2
            if x.shape[2]==40:
                stride=16
                index=20*4*20*4
            if x.shape[2]==80:
                stride=8
                index=0
            feature=x.reshape(1,65,-1)
            
            # outputs += self._process(feature, keypoints, index, x.shape[3], x.shape[2], stride)
            if use_fast:
                outputs += self._process_fast(feature, keypoints, index, x.shape[3], x.shape[2], stride)
            else:
                outputs += self._process(feature, keypoints, index, x.shape[3], x.shape[2], stride)
            
        predboxes = self._nms(outputs)
        
        # 坐标映射回原图并可视化
        vis_img = img_bgr.copy() if need_draw else None
        
        for box in predboxes: 
            # 复原像素坐标
            box.xmin = int((box.xmin - offset_x)/aspect_ratio)
            box.ymin = int((box.ymin - offset_y)/aspect_ratio)
            box.xmax = int((box.xmax - offset_x)/aspect_ratio) 
            box.ymax = int((box.ymax - offset_y)/aspect_ratio)
            
            classId = box.classId
            score = box.score
            
            title = f'{CLASSES[classId]} {score:.2f}'
            ptext =  (box.xmin, box.ymin)

            if need_draw:
                cv2.rectangle(vis_img, (box.xmin, box.ymin), (box.xmax, box.ymax), (0,255,0), 1)
                cv2.putText(vis_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

                keypoints = box.keypoint.reshape(-1,3)  
                keypoints[...,0]=(keypoints[...,0]-offset_x)/aspect_ratio
                keypoints[...,1]=(keypoints[...,1]-offset_y)/aspect_ratio
                
                for k, (x, y, conf) in enumerate(keypoints):
                    color_k = [int(x) for x in kpt_color[k]]
                    
                    if x==0 or y==0: continue
                    cv2.circle(vis_img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)                   
                
                for k, sk in enumerate(skeleton):
                    pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
                    pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))
                    
                    conf1 = keypoints[(sk[0] - 1), 2]
                    conf2 = keypoints[(sk[1] - 1), 2]
                    
                    if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                            continue
                        
                    cv2.line(vis_img, pos1, pos2, [int(x) for x in limb_color[k]], thickness=1, lineType=cv2.LINE_AA)
                    
        # print(f'Found {len(predboxes)} person')
        return predboxes, vis_img



if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  required=True)
    parser.add_argument('--image',  required=True)
    parser.add_argument('--save',   default='./result.jpg')
    parser.add_argument('--target', default='rk3588')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    det = Yolov8PoseRKNN(args.model, args.target, args.device, verbose=True)
    
    img  = cv2.imread(args.image)
    boxes, vis = det.infer(img, need_draw=True)
    cv2.imwrite(args.save, vis)
    print(f'Found {len(boxes)} person, result saved to {args.save}')
    det.release()
    
    