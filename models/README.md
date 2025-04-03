如果设备运行报错提示rknn_runtime版本问题，需要更新so
将当前目录下librknnrt.so 替换到/usr/lib

rknn, pyton3.10 模型运行或者转换需要相关的环境：
1、`pip install -r arm64_requirements_cp310.txt`
2、`pip install rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

