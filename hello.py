import cv2


def run_usb_cam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("无法打开USB摄像头, 请检查video index")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧。正在退出...")
            break


        frame = cv2.flip(frame, 1)

        cv2.imshow('usb CAM detect', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_usb_cam(0)




