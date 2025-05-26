# my_rknn_face_analysis.py
import os
import glob
import cv2
import numpy as np
from .face import Face
from .face_detect_rknn import face_detect, load_rknn_model
from .face_feature_rknn import face_feature, face_align_cv2
from .face_landmark_rknn import get_face_landmark_106
from .face_rknn_utils import draw_colored_landmarks


class MyRknnFaceAnalysis:
    def __init__(self, name='buffalo_s', root='./models', allowed_modules=None, int_8=False):
        self.name = name
        self.root = os.path.join(os.path.expanduser(root), 'models')
        self.allowed_modules = allowed_modules if allowed_modules else ['detection', 'recognition']
        self.model_paths = {}
        self.models = {}
        self.int_8 = int_8

    def prepare(self, ctx_id=-1, det_size=(640, 640), threshold=0.5):
        self.det_size = det_size
        self.det_thresh = threshold
        base_path = os.path.join(self.root, self.name)
        all_models = glob.glob(os.path.join(base_path, '*.rknn'))

        if self.int_8:
            model_map = {
                'detection': 'det_500m_int8.rknn',
                'recognition': 'w600k_mbf_int8.rknn',
                'landmark_2d_106': '2d106det.rknn'
            }
        else:
            model_map = {
                'detection': 'det_500m.rknn',
                'recognition': 'w600k_mbf.rknn',
                'landmark_2d_106': '2d106det.rknn'
            }

        for task in self.allowed_modules:
            keyword = model_map.get(task)
            if keyword is None:
                continue
            matched = [m for m in all_models if keyword in os.path.basename(m)]
            if matched:
                self.model_paths[task] = matched[0]
                self.models[task] = load_rknn_model(matched[0], target='rk3588')
                print(f"[INFO] Loaded {task} model: {matched[0]}")
            else:
                raise ValueError(f"[ERROR] Cannot find model for task: {task} in {base_path}")

    def get(self, img, max_num=0):
        faces = []
        det_model = self.models.get('detection')
        if det_model is None:
            raise RuntimeError("Detection model not loaded.")

        bboxes, kpss = face_detect(img, 
                                   input_size=self.det_size, 
                                   max_num=max_num, 
                                   det_rknn=det_model, 
                                   use_kps=True, 
                                   threshold=self.det_thresh)
        
        if bboxes is None or bboxes.shape[0] == 0:
            print("No faces detected!")
            return []

        # print("Detectded face num:", len(bboxes))
        # print("Bounding boxes:\n", bboxes)
        
        for i in range(len(bboxes)):
            face = Face()
            face.bbox = bboxes[i, :4]
            face.det_score = bboxes[i, 4]
            face.kps = kpss[i] if kpss is not None else None

            # embedding
            if 'recognition' in self.models and face.kps is not None:
                aligned = face_align_cv2(img, face.kps, (112, 112))
                features = face_feature([aligned], self.models['recognition'])
                face.embedding = features[0][0]

            # 106 landmark
            if 'landmark_2d_106' in self.models:
                lmk = get_face_landmark_106(img, face.bbox, landmark_rknn=self.models['landmark_2d_106'])
                face.landmark_2d_106 = lmk

            faces.append(face)

        return faces

    def draw_on(self, img, faces):
        dimg = img.copy()
        for i, face in enumerate(faces):
            box = face.bbox.astype(np.int32)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"Face {i+1} ({face.det_score:.2f})"
            cv2.putText(dimg, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if face.kps is not None:
                kps = face.kps.astype(np.int32)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        return dimg

    def draw_on_landmark106(self, img, faces):
        dimg = img.copy()
        for i, face in enumerate(faces):
            if face.landmark_2d_106 is not None:
                lmk = np.round(face.landmark_2d_106).astype(int)
                draw_colored_landmarks(dimg, lmk)
                # for (x, y) in np.round(face.landmark_2d_106).astype(np.int32):
                #     cv2.circle(dimg, (x, y), 1, (0, 0, 255), -1)
        return dimg

    def release(self):
        for model in self.models.values():
            model.release()
