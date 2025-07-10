import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
import time

import sys
from pathlib import Path

# 初始化 GStreamer, 必须调一次
Gst.init(None)

pipeline_str = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,width=640,height=480 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink max-buffers=1 drop=true"
)


pipeline = Gst.parse_launch(pipeline_str)

appsink = pipeline.get_by_name("sink")
appsink.set_property("emit-signals", False)
appsink.set_property("sync", False)

pipeline.set_state(Gst.State.PLAYING)

print("🚀 摄像头已启动，按 'q' 退出")
# --- 新增: FPS 计算变量 ---
pTime = 0
cTime = 0
while True:
    sample = appsink.emit("pull-sample")
    if sample is None:
        continue

    buf = sample.get_buffer()
    caps = sample.get_caps()
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    try:
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))

        frame =  cv2.flip(frame, 1)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime =  cTime

        cv2.putText(frame, f'FPS: {int(fps)}', (40, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    finally:
        buf.unmap(mapinfo)

pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
