import cv2
import cv2 as cv
import pyzbar.pyzbar as pbar
from time import sleep
import numpy as np
import pyzbar.pyzbar as pyzbar
#from PIL import Image,ImageEnhance

def decode(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)#首先使用 OpenCV 库中的 threshold 函数对灰度图像进行二值化处理，得到黑白图像
    barcodes = pyzbar.decode(gray)#然后使用 PyZbar 库中的 decode 函数对黑白图像进行二维码的解码，得到一组二维码信息
#方案一
    qrcodereturntext='Unrecognized'
    #首先定义一个字符串变量 qrcodereturntext，用于存储二维码的识别结果。
    #然后定义一个阈值变量 thre，初始值为 35，用于控制二值化操作的阈值。
    #通过一个循环来不断尝试不同的阈值，直到找到二维码为止
    thre = 35 
    while (len(barcodes) == 0 and thre < 200):                  
        ret, thresh = cv.threshold(gray, thre, 255, cv.THRESH_BINARY)#对灰度图像进行阈值操作得到二值图像。
        barcodes = pyzbar.decode(thresh)
        thre = thre + 10
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        qrcodereturntext=barcode.data
        qrcodereturntext.decode('utf-8') 
        #qrcodereturntext得到的是b类型的bytes字节数据，将其转换为字符
        print(qrcodereturntext)
        # 最后，使用一个 for 循环遍历二维码信息，获取二维码的位置和数据，
        # 并将数据存储到 qrcodereturntext 变量中。
        # 最后，将 qrcodereturntext 变量中的二维码数据输出到控制台，
        # 并将其转换为字符类型。
#方案二
    if barcodes == None:
        print("Error decoding barcodes")
    else:
        for barcode in barcodes:
            # 提取并绘制图像中条形码的边界框
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            #使用一个 for 循环遍历所有识别到的条形码。
            # 在循环内部，首先获取条形码的位置信息，并使用 OpenCV 库中的 rectangle 函数在图像上绘制出条形码的边界框。
            # 然后，使用 decode 函数将条形码的数据解码为 utf-8 编码格式，
            # 并将结果存储在 barcodeData 变量中
            print("识别成功！\n识别结果是:" ,end="")
            print(barcodeData)
            #barcodeType = barcode.type
            # # 绘出图像上条形码的数据和条形码类型
            text = "The identification result is:" + barcodeData
        # text2 = "Type:" + barcodeType
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,text,(30,30), font, 0.8,(0,255,0),2)#putText 函数将识别结果绘制在图像上。
        # cv2.putText(image,text2,(30,80), font, 0.8,(0,255,0),2)
        return image

def main_part():
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    interval = 0.1
    jk = 0
    while True:
        # Capture a frame from the camera
        f, frame = camera.read()             

        if f == True:
            # cv2.imwrite(f'D:/git/code/project/body/barcodeScaner/result{jk}.jpg', frame)     # 将图片保存为本地文件
            # img = Image.open(f'D:/git/code/project/body/barcodeScaner/result{jk}.jpg')
            # frame = cv2.imread(f'D:/git/code/project/body/barcodeScaner/result{jk}.jpg')    #读取图片
            #gray = cv2.cvtColor(frame, 0)
            # Decode the barcode from the frame 解码
            #barcodes = pbar.decode(gray)
            img = decode(frame)#调用函数
            # img = ImageEnhance.Brightness(img).enhance(2.0)#增加亮度
            # img = ImageEnhance.Sharpness(img).enhance(17.0)#锐利化
            # img = ImageEnhance.Contrast(img).enhance(4.0)#增加对比度
            # img = img.convert('L')#灰度化
            
            img = cv2.resize(img,(600,600),interpolation=cv2.INTER_CUBIC)
            cv2.imshow('camera', img)
            #删除垃圾
            # os.remove(f"D:/git/code/project/body/barcodeScaner/result{jk}.jpg")
            # #index++
            #
            # jk += 1

            #  Display the frame
            # cv2.imshow('Barcode Scanner', frame)
            #sleep(interval)

            # Check for key press
            key = cv2.waitKey(1)
            if key == 27:
                break
        else: 
            print("Capture failed")
            break
        

    #print(f"一共截下{jk}张照片,即检查产品{jk}个。")
    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_part()