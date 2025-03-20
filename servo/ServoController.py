#-*- coding: UTF-8 -*-

from PCA9685 import PCA9685
import time

# 600 ~~ 0 度
# 2400 ~~ 180度
# 1500 ~~ 90 
# index = 0,1 代表舵机位， 600 ~2400 脉冲区间，转换成对应角度 
# 
# "Sets the Servo Pulse,The PWM frequency must be 50HZ"

# 舵机通道
TILT_CHANNEL = 0  # 垂直方向舵机
PAN_CHANNEL = 1  # 水平方向舵机

MAX_ANGLE = 180
MIN_ANGLE = 0

MIN_PULSE = 600
MAX_PULSE = 2400

class ServoController:
    def __init__(self, address=0x40, freq=50, step=10, debug=False, max_puLse=MAX_PULSE, min_pulse=MIN_PULSE):
        """
        初始化舵机控制器
        :param address: PCA9685 的 I2C 地址
        :param freq: PWM 信号频率
        """
        self.pwm = PCA9685(address, debug=debug)
        self.pwm.setPWMFreq(freq)
        
        # 摆动逻辑相关
        self.current_pulse = 1500  # 初始脉宽对应 90 度
        self.step = step
        self.max_pulse = max_puLse
        self.min_pulse = min_pulse

    def angle_to_pulse(self, angle):
        return 600 + (angle / 180.0) * 1800

    def set_servo_angle(self, channel, angle):
        angle = max(0, min(180, angle))
        pulse = self.angle_to_pulse(angle)
        self.pwm.setServoPulse(channel, pulse)

    def reset_servo(self):
        """
        重置舵机到 90 度
        """
        self.set_servo_angle(TILT_CHANNEL, 90)
        self.set_servo_angle(PAN_CHANNEL, 90)

    def set_servo_pulse(self, channel, pulse):
        self.pwm.setServoPulse(channel, pulse)
        
    def swing_servo(self, channel):
        """
        自动摆动舵机：在最小和最大脉宽之间往复运动。
        """
        try:
            while True:
                self.pwm.setServoPulse(channel, self.current_pulse)
                time.sleep(0.03)
                self.current_pulse += self.step
                if self.current_pulse >= self.max_pulse or self.current_pulse <= self.min_pulse:
                    self.step = -self.step  # 到达边界后反向摆动
        finally:
            print("舵机旋转完成")



if __name__ == '__main__':
    controller = ServoController(debug=True)


    controller.reset_servo()



