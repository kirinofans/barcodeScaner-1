import cv2
import pyzbar.pyzbar as pbar
from time import sleep
import numpy as np
import os
import argparse
import imutils

def main_part():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    interval = 0.5
    jk = 0

    while True:
        # Capture a frame from the camera
        f, frame = cap.read()               # 将摄像头中的一帧图片数据保存
        cv2.imwrite(f'D:/git/code/project/body/barcodeScaner/result{jk}.jpg', frame)     # 将图片保存为本地文件
        frame = cv2.imread(f'D:/git/code/project/body/barcodeScaner/result{jk}.jpg')    #读取图片
        gray = cv2.cvtColor(frame, 0)
        # Decode the barcode from the frame 解码
        barcodes = pbar.decode(gray)

        if barcodes == []:
            print("Error decoding barcodes")
        else:
            print("识别成功")
            for con in barcodes:
                barcodes_data = con.data.decode("utf-8")
                print("识别结果是:" ,end="")
                print(barcodes_data)


        #删除垃圾
        os.remove(f"D:/git/code/project/body/barcodeScaner/result{jk}.jpg")
        #index++
        jk += 1

        #  Display the frame
        cv2.imshow('Barcode Scanner', frame)
        sleep(interval)

        # Check for key press
        key = cv2.waitKey(1)
        if key == 27:
            break

    print(f"一共截下{jk}张照片,即检查产品{jk}个。")
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # todo 修改为你要进行检测的图片路径即可
    main_part()