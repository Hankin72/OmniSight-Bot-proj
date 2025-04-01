# -*- coding: utf-8 -*-
# models/MyRknnFaceAnalysis.py
# Encapsulates RKNN-based face analysis pipeline (detection, landmark, recognition)

import cv2
import numpy as np
import os.path as osp
import glob

# Import necessary components from other modules in this package
try:
    from .face_detect_rknn import load_rknn_model as load_rknn, face_detect
    from .face_landmark_api import FaceLandmarkDetector # Use the class created earlier
    from .face_feature_rknn import face_align_cv2, face_feature
    from .face_align import transform, trans_points # Might be needed internally by landmark/align
except ImportError:
    print("Error: Running MyRknnFaceAnalysis directly or imports failed.")
    # Add fallbacks if direct execution is needed, otherwise rely on package structure
    from face_detect_rknn import load_rknn_model as load_rknn, face_detect
    from face_landmark_api import FaceLandmarkDetector
    from face_feature_rknn import face_align_cv2, face_feature
    from face_align import transform, trans_points

# Define a simple Face class to hold results, similar to insightface's common.Face
class Face:
    """
    Represents a detected face and its associated attributes.
    """
    def __init__(self, bbox=None, kps=None, det_score=None, landmark_106=None, embedding=None, aligned_face=None):
        self.bbox = bbox             # Bounding box [x1, y1, x2, y2]
        self.kps = kps               # 5 keypoints from detection [5, 2]
        self.det_score = det_score   # Detection score
        self.landmark_106 = landmark_106 # 106 keypoints from landmark model [106, 2 or 3]
        self.embedding = embedding   # Feature embedding vector [typically 512,]
        self.aligned_face = aligned_face # The 112x112 aligned face image used for feature extraction

    def __repr__(self):
        return (f"Face(bbox={self.bbox}, kps={self.kps is not None}, "
                f"det_score={self.det_score:.4f}, landmark_106={self.landmark_106 is not None}, "
                f"embedding={self.embedding is not None})")


class MyRknnFaceAnalysis:
    """
    RKNN-based Face Analysis Pipeline.
    Loads detection, landmark, and recognition models and provides methods
    to process images.
    """
    def __init__(self, det_model_path, lmk_model_path, rec_model_path, target='rk3588'):
        """
        Initializes the analysis pipeline by loading RKNN models.

        Args:
            det_model_path (str): Path to the detection RKNN model (e.g., det_500m.rknn).
            lmk_model_path (str): Path to the 106-landmark RKNN model (e.g., 2d106det.rknn).
            rec_model_path (str): Path to the recognition RKNN model (e.g., w600k_mbf.rknn).
            target (str): Target NPU platform (e.g., 'rk3588').
        """
        print("Initializing MyRknnFaceAnalysis...")
        self.target = target
        self.det_rknn = None
        self.landmark_detector = None
        self.rec_rknn = None
        self.det_size = (640, 640) # Default detection size
        self.det_thresh = 0.5      # Default detection threshold

        # Load Detection Model
        print(f"Loading Detection model: {det_model_path}")
        self.det_rknn = load_rknn(det_model_path, target=self.target)
        if self.det_rknn is None:
            raise RuntimeError(f"Failed to load detection model: {det_model_path}")
        print("Detection model loaded.")

        # Load Landmark Model (using the previously created class)
        print(f"Loading Landmark model: {lmk_model_path}")
        try:
            # Assuming FaceLandmarkDetector handles its own loading via __init__
            self.landmark_detector = FaceLandmarkDetector(lmk_model_path, target=self.target)
        except Exception as e:
            raise RuntimeError(f"Failed to load landmark model: {lmk_model_path} - {e}")
        print("Landmark model loaded.")

        # Load Recognition Model
        print(f"Loading Recognition model: {rec_model_path}")
        self.rec_rknn = load_rknn(rec_model_path, target=self.target)
        if self.rec_rknn is None:
            raise RuntimeError(f"Failed to load recognition model: {rec_model_path}")
        print("Recognition model loaded.")

        print("MyRknnFaceAnalysis initialized successfully.")

    def prepare(self, det_thresh=0.5, det_size=(640, 640)):
        """
        Sets detection parameters.

        Args:
            det_thresh (float): Detection threshold.
            det_size (tuple): Detection input size (width, height).
        """
        self.det_thresh = det_thresh
        self.det_size = det_size
        print(f"Set det_thresh={self.det_thresh}, det_size={self.det_size}")
        # In the future, if individual models need preparation, call their prepare methods here.

    def get(self, img, max_num=0):
        """
        Performs the full face analysis pipeline on an image.

        Args:
            img (np.ndarray): Input image (BGR format).
            max_num (int): Maximum number of faces to detect (0 for all).

        Returns:
            list[Face]: A list of Face objects containing analysis results.
        """
        if self.det_rknn is None or self.landmark_detector is None or self.rec_rknn is None:
            print("Error: Models not fully loaded.")
            return []

        # 1. Detection
        # Note: conf_thres seems not to be an argument for face_detect; thresholding might be internal.
        bboxes, kpss_5 = face_detect(img,
                                     input_size=self.det_size,
                                     # conf_thres=self.det_thresh, # Removed unexpected argument
                                     max_num=max_num,
                                     det_rknn=self.det_rknn,
                                     use_kps=True) # Ensure 5kps are returned

        if bboxes is None or bboxes.shape[0] == 0:
            return [] # No faces detected

        results = []
        aligned_faces_batch = []
        face_indices_for_rec = [] # Keep track of which faces will get embeddings

        # Prepare batch for recognition
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps5 = kpss_5[i] if kpss_5 is not None else None

            face = Face(bbox=bbox, kps=kps5, det_score=det_score)

            # 2. Landmark Detection (per face)
            if self.landmark_detector and kps5 is not None: # Use bbox for landmark
                 landmarks_106 = self.landmark_detector.get_landmarks(img, bbox)
                 face.landmark_106 = landmarks_106

            # 3. Alignment (prepare for recognition)
            if self.rec_rknn and kps5 is not None:
                # Use the correct parameter name 'image_size'
                aligned_face = face_align_cv2(img, kps5, image_size=(112, 112))
                if aligned_face is not None:
                    aligned_faces_batch.append(aligned_face)
                    face.aligned_face = aligned_face # Store for potential later use
                    face_indices_for_rec.append(i) # Mark this face for embedding update

            results.append(face) # Add face even if alignment/rec fails

        # 4. Recognition (batch inference)
        if self.rec_rknn and aligned_faces_batch:
            embeddings = face_feature(aligned_faces_batch, self.rec_rknn)
            if embeddings is not None and len(embeddings) == len(face_indices_for_rec):
                for idx, embedding in zip(face_indices_for_rec, embeddings):
                    results[idx].embedding = embedding # Update the correct Face object

        return results

    def release(self):
        """Releases all loaded RKNN model resources."""
        print("Releasing MyRknnFaceAnalysis resources...")
        if self.det_rknn:
            self.det_rknn.release()
            self.det_rknn = None
            print("Detection model released.")
        if self.landmark_detector:
            self.landmark_detector.release() # Assumes FaceLandmarkDetector has a release method
            self.landmark_detector = None
            # print("Landmark model released.") # Printed by FaceLandmarkDetector.release()
        if self.rec_rknn:
            self.rec_rknn.release()
            self.rec_rknn = None
            print("Recognition model released.")
        print("MyRknnFaceAnalysis resources released.")

# Example Usage
if __name__ == '__main__':
    # Define model paths (adjust as necessary)
    BASE_MODEL_DIR = "/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s"
    DET_MODEL = osp.join(BASE_MODEL_DIR, "det_500m.rknn")
    LMK_MODEL = osp.join(BASE_MODEL_DIR, "2d106det.rknn")
    REC_MODEL = osp.join(BASE_MODEL_DIR, "w600k_mbf.rknn")

    # Image path
    IMG_PATH = "/home/orangepi/Documents/new_face_algorithm_proj/dataset/01_IMG_1029.JPG"

    # Check if models exist
    if not all(osp.exists(p) for p in [DET_MODEL, LMK_MODEL, REC_MODEL]):
         print("Error: One or more model files not found. Check paths.")
         print(f"DET: {DET_MODEL} - Exists: {osp.exists(DET_MODEL)}")
         print(f"LMK: {LMK_MODEL} - Exists: {osp.exists(LMK_MODEL)}")
         print(f"REC: {REC_MODEL} - Exists: {osp.exists(REC_MODEL)}")
         exit()

    print("Starting RKNN Face Analysis Example...")
    analyzer = None
    try:
        # Initialize
        analyzer = MyRknnFaceAnalysis(det_model_path=DET_MODEL,
                                      lmk_model_path=LMK_MODEL,
                                      rec_model_path=REC_MODEL)

        # Prepare (optional, sets defaults if not called)
        analyzer.prepare(det_thresh=0.5, det_size=(640, 640))

        # Load image
        img_bgr = cv2.imread(IMG_PATH)
        if img_bgr is None:
            print(f"Error: Failed to load image {IMG_PATH}")
            exit()

        print(f"Processing image: {IMG_PATH}")
        # Get analysis results
        faces = analyzer.get(img_bgr)

        print(f"\nFound {len(faces)} faces.")
        for i, face in enumerate(faces):
            print(f"--- Face {i+1} ---")
            print(f"  Detection Score: {face.det_score:.4f}")
            print(f"  Bounding Box: {face.bbox}")
            print(f"  5 Keypoints detected: {face.kps is not None}")
            print(f"  106 Landmarks detected: {face.landmark_106 is not None}")
            if face.landmark_106 is not None:
                 print(f"    Landmark shape: {face.landmark_106.shape}")
            print(f"  Embedding calculated: {face.embedding is not None}")
            if face.embedding is not None:
                 print(f"    Embedding shape: {face.embedding.shape}, first 5 dims: {face.embedding[:5]}")

        # Simple visualization (optional)
        img_draw = img_bgr.copy()
        for face in faces:
             # Draw bbox
             box = face.bbox.astype(np.int32)
             cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
             # Draw 5 kps
             if face.kps is not None:
                 for kx, ky in face.kps.astype(np.int32):
                     cv2.circle(img_draw, (kx, ky), 2, (255, 0, 0), -1) # Blue
             # Draw 106 landmarks
             if face.landmark_106 is not None:
                 for lx, ly in face.landmark_106[:, :2].astype(np.int32): # Only draw x,y
                     if 0 <= lx < img_draw.shape[1] and 0 <= ly < img_draw.shape[0]:
                         cv2.circle(img_draw, (lx, ly), 1, (0, 0, 255), -1) # Red

        print("\nDisplaying image with detections. Press any key to close.")
        cv2.imshow("RKNN Face Analysis", img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        if analyzer:
            analyzer.release()

    print("RKNN Face Analysis Example Finished.")
