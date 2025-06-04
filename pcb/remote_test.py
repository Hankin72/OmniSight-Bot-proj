import RPi.GPIO as GPIO
import time
import Pcb_Car

PIN = 36
buzzer = 32

GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)

ir_repeat_cnt = 0

car = Pcb_Car.Pcb_Car()

def init():
    GPIO.setup(PIN, GPIO.IN, GPIO.PUD_UP)
    GPIO.setup(buzzer, GPIO.OUT)

    print("IR test start...") 


def whistle():
    p = GPIO.PWM(buzzer, 400)
    p.start(50)
    time.sleep(0.5)
    p.stop()
    
    
    
def exec_cmd(key_val):
    if key_val==0x45:  # power
        car.Ctrl_Servo(1, 90)
        car.Ctrl_Servo(2, 90)
        car.Car_Stop()
    elif key_val == 0x40:  # button  +
        car.Car_Run(50, 50)  # move forward
    elif key_val == 0x15:  # BUTTON pause
        car.Car_Stop() 
        
    elif key_val == 0x07:  # button  << 
        car.Car_Left(50, 50) # left 
        
    elif key_val == 0x47:  # MENU button
        whistle()
        
    elif key_val == 0x09:  # button >>
        car.Car_Right(50, 50) 
        
    elif key_val == 0x16:  # button 0
        car.Car_Spin_Left(50, 50) 
        
    elif key_val == 0x19:  # button -
        car.Car_Back(50, 50) 
        
    elif key_val == 0x0d:  # button c
        car.Car_Spin_Right(50, 50)
        
    elif key_val == 0x0c:  # button 1
        car.Ctrl_Servo(1, 0) 
        
    elif key_val == 0x18:  # button 2
        car.Ctrl_Servo(1, 90) 
        
    elif key_val == 0x5e:  # button 3
        car.Ctrl_Servo(1, 180) 

    elif key_val == 0x08:  # button 4
        car.Ctrl_Servo(2, 0) 
        

    elif key_val == 0x1c:  # button 5
        car.Ctrl_Servo(2, 90) 

    elif key_val == 0x5a:  # button  6
        car.Ctrl_Servo(2, 180) 

    else:
        print(key_val)
        print("no cmd")

try:
    init()
    while True:
        # 等待红外起始低电平
        if GPIO.input(PIN) == 0:
            ir_repeat_cnt = 0
            count = 0
            while GPIO.input(PIN) == 0 and count < 200:      # 9 ms 低电平
                count += 1
                time.sleep(0.00006)

            count = 0
            while GPIO.input(PIN) == 1 and count < 80:       # 4.5 ms 高电平
                count += 1
                time.sleep(0.00006)

            idx = 0
            cnt = 0
            data = [0, 0, 0, 0]

            for i in range(32):                              # 32 bit 数据
                count = 0
                while GPIO.input(PIN) == 0 and count < 15:   # 560 µs 低电平
                    count += 1
                    time.sleep(0.00006)

                count = 0
                while GPIO.input(PIN) == 1 and count < 40:   # 判断是 0 还是 1
                    count += 1
                    time.sleep(0.00006)

                if count > 9:                                # 逻辑 1
                    data[idx] |= 1 << cnt

                if cnt == 7:                                 # 下一个字节
                    cnt = 0
                    idx += 1
                else:
                    cnt += 1

            # 校验和：低 8 位 + 高 8 位 == 0xFF
            if data[0] + data[1] == 0xFF and data[2] + data[3] == 0xFF:
                print("Get the key: 0x%02x" % data[2])
                exec_cmd(data[2])

        else:
            # 判断红外遥控接收器是否松开，但为复现瞬时时间约 11 ms，所以这里逻辑是读取 110 × 0.001
            if ir_repeat_cnt > 110:
                ir_repeat_cnt = 0
                car.Car_Stop()
            else:
                time.sleep(0.001)
                ir_repeat_cnt += 1

except KeyboardInterrupt:
    pass

print("Ending")
GPIO.cleanup()
