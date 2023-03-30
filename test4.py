import cv2
import cv2 as cv
import pyzbar.pyzbar as pbar
from time import sleep
import numpy as np
import pyzbar.pyzbar as pyzbar
def rotate_image(image, angle):
    # 计算旋转矩阵
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # 旋转图像
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def barcodee(gray):
    # 尝试直接解码
    texts = pyzbar.decode(gray)
    if texts != []:
        return texts

    # 尝试旋转后解码
    angle = barcode_angle(gray)
    if angle < -45:
        angle = -90 - angle
    rotated_gray = rotate_image(gray, angle)
    texts = pyzbar.decode(rotated_gray)
    if texts != []:
        return texts

    # 尝试增强对比度后解码
    enhanced_gray = np.uint8(np.clip((1.1 * gray + 10), 0, 255))
    angle = barcode_angle(enhanced_gray)
    if angle < -45:
        angle = -90 - angle
    rotated_enhanced_gray = rotate_image(enhanced_gray, angle)
    texts = pyzbar.decode(rotated_enhanced_gray)
    return texts

def bar(image, angle):
    # 将输入图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 尝试旋转后解码
    rotated_gray = rotate_image(gray, 0 - angle)
    texts = pyzbar.decode(rotated_gray)
    if texts != []:
        return texts

    # 尝试增强对比度后解码
    enhanced_gray = np.uint8(np.clip((1.1 * gray + 10), 0, 255))
    rotated_enhanced_gray = rotate_image(enhanced_gray, 0 - angle)
    texts = pyzbar.decode(rotated_enhanced_gray)
    return texts

def barcode_angle(image):
    # 将输入图像转换为灰度图像
    ret, gray = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)

    # 进行二值化处理
    threshold_value, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # 对二值图像进行膨胀和腐蚀操作
    kernel = np.ones((8, 8), np.uint8)
    erosion = cv2.erode(cv2.dilate(binary, kernel, iterations=1), kernel, iterations=3)

    # 进行轮廓检测
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 计算最小外接矩形的角度
    if len(contours) == 0:
        angle = 0
    else:
        rect = cv2.minAreaRect(contours[0])
        angle = rect[2]

    return angle
def rotate_bound(image, angle):
  #首先，定义了一个名为 rotate_bound 的函数，它有两个输入参数，
  # 分别是 image 和 angle，其中 image 表示输入的图像，angle 表示要旋转的角度。
  # 函数的返回值是一个经过旋转后的图像。
  (h, w) = image.shape[:2]
  (cX, cY) = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  #使用 OpenCV 库中的 getRotationMatrix2D 函数生成旋转矩阵 M，
  # 其中 M 的第一个参数是旋转中心点的坐标，第二个参数是旋转的角度，第三个参数是缩放系数。
  # 这里的缩放系数为 1.0，表示不进行缩放。
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])
  #西瓜6写的，转载需声明
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))
  #接着，根据旋转矩阵 M 计算出旋转后的图像的宽度和高度，并将它们转换为整数类型。
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY
  #然后，根据旋转后的图像的宽度和高度，将旋转中心点的坐标进行调整，使得旋转后的图像不会超出原图的范围。
  return cv2.warpAffine(image, M, (nW, nH))#使用 OpenCV 库中的 warpAffine 函数对输入图像进行旋转操作，并返回旋转后的图像。

def decode(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    texts = barcodee(gray)
    # print(texts)
    if texts==[]:
        print("未识别成功")
    else:
        for text in texts:
            tt = text.data.decode("utf-8")
            print("识别成功")
            print(tt)  
    barcodes = pyzbar.decode(gray)
    if barcodes == None:
        print("Error decoding barcodes")
    else:
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            print("识别成功！\n识别结果是:" ,end="")
            print(barcodeData)
        return image

def main_part():
    camera = cv2.VideoCapture(0)
    interval = 0.1
    while True:
        f, frame = camera.read()   
        if f == True:
            img = decode(frame)#调用函数
            img = cv2.resize(img,(600,600),interpolation=cv2.INTER_CUBIC)
            cv2.imshow('camera', img)
            key = cv2.waitKey(1)
            if key == 27:
                break
            sleep(interval)
        else: 
            print("Capture failed")
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main_part()