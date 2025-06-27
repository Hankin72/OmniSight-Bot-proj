from time import perf_counter
from rknn.api import RKNN
import numpy as np, cv2

rknn = RKNN()
rknn.load_rknn('yolov8n-pose_int8.rknn')
rknn.init_runtime(target='rk3588')
# rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_0_1_2)

img = np.zeros((640,640,3),dtype=np.uint8)  # 纯黑，排除字节拷贝
start = perf_counter()
for _ in range(200):
    rknn.inference(inputs=[img])

fps = 200 / (perf_counter()-start)

print('NPU-only FPS:', fps)
