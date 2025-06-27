
## 安装（Debian/Ubuntu/Armbian）

`sudo apt-get install v4l-utils`

## 列出所有格式-分辨率-帧率

`v4l2-ctl -d /dev/video0 --list-formats-ext`