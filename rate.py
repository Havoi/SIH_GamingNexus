import time, serial

ser = serial.Serial("COM5",115200)
count = 0
t0 = time.time()

while True:
    ser.readline()
    count+=1
    if time.time()-t0 >= 1.0:
        print("Hz =", count)
        count=0
        t0=time.time()