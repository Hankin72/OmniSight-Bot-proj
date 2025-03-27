#-*- coding: UTF-8 -*-

import time

# 定义 PID 控制器类，用于计算舵机角度调整量
class PIDController:
    def __init__(self, kp=0.5, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        # 计算时间差
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:  # 避免除以0
            dt = 1e-6
            
        # 计算积分和微分项
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        # 计算 PID 输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 保存此次误差和时间供下次计算
        self.last_error = error
        self.last_time = current_time
        return output