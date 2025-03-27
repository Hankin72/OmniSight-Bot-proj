# onnx_to_rknn_3588.py

from rknn.api import rknn, RKNN
import numpy as np

# RKNN model config
ONNX_MODEL = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/det_500m.onnx'
RKNN_MODEL = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/det_500m.rknn'


# ONNX_MODEL = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/w600k_mbf.onnx'
# RKNN_MODEL = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/w600k_mbf.rknn'


# ONNX_MODEL = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/2d106det.onnx'
# RKNN_MODEL = '/home/orangepi/Documents/face_algorithm_proj/models/models/buffalo_s/2d106det.rknn'

PLATFORM = "rk3588"


# Create RKNN object
rknn = RKNN(verbose=True)

# pre-process config
print('--> Config model')
# rknn.config(target_platform=PLATFORM)
rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[128, 128, 128]], target_platform=PLATFORM)
print('done')


# Load ONNX model
print('--> Loading model')
# ret = rknn.load_onnx(model=ONNX_MODEL, inputs=['input'], input_size_list=[[1, 30, 128]])


# ret = rknn.load_onnx(model=ONNX_MODEL,inputs=['input.1'],  input_size_list=[[1, 3, 112, 112]])  #w600k_mbf.onnx

ret = rknn.load_onnx(model=ONNX_MODEL, inputs=['input.1'],  input_size_list=[[1, 3, 640, 640]])  #det_500m.onnx

# ret = rknn.load_onnx(model=ONNX_MODEL,inputs=['data'],  input_size_list=[[1, 3, 192, 192]])      #2d106det.onnx

if ret != 0:
    print('Load ONNX model failed!')
    exit(ret)
print('done')


# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')


# Export RKNN model
print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export RKNN model failed!')
    exit(ret)
print('done')

# Release RKNN context
rknn.release()