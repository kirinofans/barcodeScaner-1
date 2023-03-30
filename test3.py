import cv2
import cv2 as cv
import pyzbar.pyzbar as pbar
from time import sleep
import numpy as np
import pyzbar.pyzbar as pyzbar

def barcodee(gray):
  texts = pyzbar.decode(gray)
  if texts == []:
    angle = barcode_angle(gray)
    if angle < -45:
      angle = -90 - angle
    texts = bar(gray, angle)
  if texts == []:
    gray = np.uint8(np.clip((1.1 * gray + 10), 0, 255))
    angle = barcode_angle(gray)
    if angle < -45:
      angle = -90 - angle
    texts = bar(gray, angle)
    #调用 barcode_angle 函数计算输入图像中条形码的旋转角度，并进行修正，使条形码水平方向与图像边缘平行。
    # 调用 bar 函数对修正后的图像进行条形码解码，如果解码结果仍为空，则进行下一步操作。
    # 对输入图像进行亮度调整（增加 10 个灰度级，再乘以 1.1），
    # 然后重复步骤2和3，直到得到非空的解码结果或者达到最大尝试次数。
    # 返回解码结果。
  return texts

def bar(image, angle):
  #首先，定义了一个名为 bar 的函数，它有两个输入参数，分别是 image 和 angle，
  # 其中 image 表示输入的图像，angle 表示条形码的倾斜角度。
  #接下来，将输入图像转换为灰度图像，并调用上文提到的 rotate_bound 函数对图像进行旋转操作，
  # 将图像恢复到水平方向。旋转的角度是 0 - angle，表示将图像逆时针旋转 angle 度。
  gray = image
  bar = rotate_bound(gray, 0 - angle)
  roi = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
  texts = pyzbar.decode(roi)
  #然后，将旋转后的图像转换为 RGB 格式，并使用 PyZbar 库中的 decode 函数对图像进行条形码的解码。
  # 解码结果存储在名为 texts 的变量中。
  #最后，函数返回解码结果，即识别出的条形码信息。
  return texts

def barcode_angle(image):
  gray = image
  ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
  kernel = np.ones((8, 8), np.uint8)
  dilation = cv2.dilate(binary, kernel, iterations=1)
  erosion = cv2.erode(dilation, kernel, iterations=1)
  erosion = cv2.erode(erosion, kernel, iterations=1)
  erosion = cv2.erode(erosion, kernel, iterations=1)
  #接下来，将输入图像转换为灰度图像，
  # 并使用 OpenCV 库中的 threshold 函数对灰度图像进行二值化处理。
  # 然后，使用一个 8x8 的矩形结构元素对二值图像进行**膨胀和腐蚀**操作，
  # 得到一幅清晰的条形码图像。
  contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  #接着，使用 OpenCV 库中的 findContours 函数对处理后的图像进行轮廓检测。如果没有检测到轮廓，则返回角度值为 0。
  if len(contours) == 0:
    rect = [0, 0, 0]
  else:
    rect = cv2.minAreaRect(contours[0])
  #如果检测到了轮廓，则使用 OpenCV 库中的 minAreaRect 函数计算出轮廓的最小外接矩形。
  # 最小外接矩形是能够完全包围轮廓的最小矩形，它的长宽比能够反映出条形码的倾斜角度。
  # 这里直接返回最小外接矩形的角度值 rect[2]，即条形码的倾斜角度。
  #最后，函数返回条形码的角度值。
  return rect[2]

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
        print("未识别成功或没有识别对象")
        warning()
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
def warning():#识别失败
  pass
def main_part():
    camera = cv2.VideoCapture(0)
    interval = 0.2
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
            print("Capture failed")#摄像头读取失败
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main_part()