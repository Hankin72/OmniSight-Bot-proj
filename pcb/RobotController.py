# robot_controller.py
import time
import RPi.GPIO as GPIO
# import Pcb_Car
from .Pcb_Car import Pcb_Car

class RobotController:
    def __init__(self, trig_pin=16, echo_pin=18):
        self.car = Pcb_Car()

        # 超声波初始化
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

        # 控制参数
        self.frame_center_x = 320  # 640宽图像中心
        self.target_height = 300   # 理想bbox高度，表示"理想距离"
        self.dead_zone_x = 20
        self.dead_zone_h = 40

        self.max_speed = 60

    # ---------- 超声波测距 ----------
    def get_distance(self):
        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(self.trig_pin, GPIO.LOW)

        t_start = time.time()
        while not GPIO.input(self.echo_pin):
            if time.time() - t_start > 0.03:
                return -1

        t1 = time.time()
        while GPIO.input(self.echo_pin):
            if time.time() - t1 > 0.03:
                return -1
        t2 = time.time()

        distance_cm = ((t2 - t1) * 340 / 2) * 100
        return distance_cm

    def avoid_if_needed(self, threshold=20):
        dist = self.get_distance()
        if 0 < dist < threshold:
            self.stop()
            self.car.Car_Spin_Right(60, 60)
            time.sleep(1)
            return True
        return False

    # ---------- 主控制逻辑 ----------
    def track_target(self, offset_x, height_px):
        """
        offset_x: 目标中心点相对于图像中心的偏移量（正右负左）
        height_px: 目标 bbox 高度，用于判断远近
        """

        # 避障检测
        if self.avoid_if_needed():
            return

        # 横向控制：左右旋转
        if abs(offset_x) > self.dead_zone_x:
            if offset_x > 0:
                self.car.Car_Spin_Right(self.max_speed, self.max_speed)
            else:
                self.car.Car_Spin_Left(self.max_speed, self.max_speed)
            time.sleep(0.1)
            self.stop()
            return

        # 纵向控制：前进 / 后退
        if height_px < self.target_height - self.dead_zone_h:
            self.car.Car_Run(self.max_speed, self.max_speed)
            
        elif height_px > self.target_height + self.dead_zone_h:
            self.car.Car_Back(self.max_speed, self.max_speed)
            
        else:
            self.stop()

    def stop(self):
        self.car.Car_Stop()

    def set_servo(self, id, angle):
        self.car.Ctrl_Servo(id, angle)

    def cleanup(self):
        self.car.Car_Stop()
        GPIO.cleanup()

