import cv2
import numpy as np
import time

# Load face classifier
face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# Open webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Camera cannot be accessed")
    exit()

quit_pressed = False

pTime = 0
while not quit_pressed:
    ret, frame = video_capture.read()

    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_list = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=2)

        for (x, y, w, h) in face_list:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        text = "Faces Detected: " + str(len(face_list))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (10, 30), font, 1, (255, 0, 0), 2)

        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_pressed = True
            break

video_capture.release()
cv2.destroyAllWindows()
