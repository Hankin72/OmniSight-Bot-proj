import Pcb_Car 
import time

car = Pcb_Car.Pcb_Car()


car.Car_Run(50, 50)  # 50 的速度前进 2秒， 速度取值范围 0~255
time.sleep(2)
# car.Car_Stop()


car.Car_Back(50, 50)  # 50 的速度后退 2秒， 
time.sleep(2)
# car.Car_Stop()



car.Car_Left(0, 50)  # 50 的速度左转 2秒， 
time.sleep(2)
# car.Car_Stop()


car.Car_Right(50, 0)  # 50 的速度右转 2秒， 
time.sleep(2)
# car.Car_Stop()


car.Car_Spin_Left(50, 50)  # 50 的速度左旋 2 秒
time.sleep(5)
# car.Car_Stop()


car.Car_Spin_Right(50, 50)  # 50 的速度右旋 2 秒
time.sleep(5)
car.Car_Stop()
