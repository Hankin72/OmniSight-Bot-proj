from models.myRknnFaceAnalysis import MyRknnFaceAnalysis
import cv2
import numpy as np
from FaceUtils import draw_colored_landmarks, COLOR_GREEN, COLOR_RED
from FaceDatabase import load_face_database

DEFAULT_FILENAME_DB = 'my_face_database_s.npy'
FACE_DATABASE = load_face_database(filename=DEFAULT_FILENAME_DB)
threshold = 0.45

def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度
    参数:
    - a: 向量 a
    - b: 向量 b
    返回:
    - similarity: 余弦相似度 (介于 -1 和 1 之间)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
    
    
def recognize_face(embedding):
    if embedding is None:
        print("Error: 提取的 embedding 为 None")
        return 'Unknown'

    best_match = None
    highest_similarity = -1
    for name, embeddings in FACE_DATABASE.items():
        for known_embedding in embeddings:
            if known_embedding is None:
                print(f"Warning: {name} 的 embedding 为 None，跳过")
                continue
            similarity = cosine_similarity(embedding, known_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = name
    if highest_similarity >= threshold:
        return best_match, highest_similarity
    else:
        return 'Unknown', highest_similarity



if __name__ == "__main__":    
    img_path = "/home/orangepi/Documents/face_algorithm_proj/dataset/01_IMG_1029.JPG"

    img = cv2.imread(img_path)

    app = MyRknnFaceAnalysis(name="buffalo_s",
                            root="/home/orangepi/Documents/face_algorithm_proj/models/",
                            allowed_modules=["detection", "recognition"], int_8=True)
    
    app.prepare(det_size=(640, 640))

    target_name="hankin"

    faces = app.get(img)
    
    frame = img.copy()
    for face in faces:
        
        bbox = face.bbox.astype(int)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_GREEN, 1)
            
        embedding = face.embedding
        if embedding is None:
            print("Error: 无法提取特征向量")


        name, pre = recognize_face(embedding)
        # pre = "{:.2f}".format(pre)
        # temp_label = f"{name} {pre}"
        # print(temp_label)
        if target_name and name == target_name:
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
        else:
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)

        # for face in faces:
        #     bbox = face.bbox.astype(int)
            
        #     lmk = face.landmark_2d_106
        #     lmk = np.round(lmk).astype(int)

        #     draw_colored_landmarks(img_drawn, lmk)

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
