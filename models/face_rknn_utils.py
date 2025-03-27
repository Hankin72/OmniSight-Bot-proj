import matplotlib.pyplot as plt
import cv2


def showimage(image, figsize=(13, 13)):
    # 创建指定大小的画布
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()