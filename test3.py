import cv2
import cv2 as cv
import pyzbar.pyzbar as pbar
from time import sleep
import numpy as np
import pyzbar.pyzbar as pyzbar

def barcodee(gray):
  texts = pyzbar.decode(gray)
  # if texts == []:
  #   angle = barcode_angle(gray)
  #   if angle < -45:
  #     angle = -90 - angle
  #   texts = bar(gray, angle)
  if texts == []:
    gray = np.uint8(np.clip((1.1 * gray + 10), 0, 255))
    #np.clip函数将所有小于下边界的数值全部改为下边界， 将大于上边界的数值全部改为上边界。
    #此处0是下界255是上界
    #np.uint8范围是0~255
    angle = barcode_angle(gray)
    if angle < -45:
      angle = -90 - angle
    texts = bar(gray, angle)
    #调用 barcode_angle 函数计算输入图像中条形码的旋转角度，并进行修正，使条形码水平方向与图像边缘平行。
    # 2.调用 bar 函数对修正后的图像进行条形码解码，如果解码结果仍为空，则进行下一步操作。
    # 3.对输入图像进行亮度调整（增加 10 个灰度级，再乘以 1.1），
    #### 然后重复步骤2和3，直到得到非空的解码结果或者达到最大尝试次数。
    # 返回解码结果。
  return texts

def bar(image, angle):
  #首先，定义了一个名为 bar 的函数，它有两个输入参数，分别是 image 和 angle，
  # 其中 image 表示输入的图像，angle 表示条形码的倾斜角度。
  #接下来，将输入图像转换为灰度图像，并调用上文提到的 rotate_bound 函数对图像进行旋转操作，
  # 将图像恢复到水平方向。旋转的角度是 0 - angle，表示将图像逆时针旋转 angle 度。
  # gray = image
  # bar = rotate_bound(gray, 0 - angle)
  # roi = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
  # texts = pyzbar.decode(roi)
  bar = rotate_bound(image, 0 - angle)
  tem = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
  results = pyzbar.decode(tem)
  #然后，将旋转后的图像转换为 RGB 格式，并使用 PyZbar 库中的 decode 函数对图像进行条形码的解码。
  # 解码结果存储在名为 texts 的变量中。
  #最后，函数返回解码结果，即识别出的条形码信息。
  return results

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
  #分别是 image 和 angle，其中 image 表示输入的图像，angle 表示要旋转的角度。
  #函数的返回值是一个经过旋转后的图像。
  (h, w) = image.shape[:2]
  # image.shape[0], 图片垂直尺寸
  # image.shape[1], 图片水平尺寸
  # image.shape[2], 图片通道数
  (cX, cY) = (w // 2, h // 2)#获取矩形中心点坐标
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)#获得一个旋转矩阵
  #使用 OpenCV 库中的 getRotationMatrix2D 函数生成旋转矩阵 M，
  #其中 M 的第一个参数是旋转中心点的坐标(cX, cY)，第二个参数是旋转的角度-angle，第三个参数是缩放系数1.0没缩放。
  
  #在计算机图形学中，为了统一将平移、旋转、缩放等用矩阵表示，需要引入齐次坐标。（假设使用​​​​​​​2×2
  #的矩阵，是没有办法描述平移操作的，只有引入​​​​​​​3×3矩阵形式，才能统一描述二维中的平移、旋转、缩放操作。同理必须使用​​​​​​​4×4的矩阵才能统一描述三维的变换）
  
  # 首先将旋转中心平移到原点
  # 按上述描述的绕原点进行旋转
  # 再将旋转中心平移回原来的位置
  #M
#   [[ 6.123234e-17  1.000000e+00  1.500000e+02]
#  [-1.000000e+00  6.123234e-17  6.500000e+02]]
  cos = np.abs(M[0, 0])#返回数组各元素绝对值
  sin = np.abs(M[0, 1])
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))
  #接着，根据旋转矩阵 M 计算出旋转后的图像的"""""""宽度和高度"""""""，并将它们转换为整数类型。
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY
  #然后，根据旋转后的图像的宽度和高度，将旋转""""中心点的坐标进行调整""""，使得旋转后的图像不会超出原图的范围。

#cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
# src – 输入图像。
# M – 变换矩阵。
# dsize – 输出图像的大小。
# flags – 插值方法的组合（int 类型！）
# borderMode – 边界像素模式（int 类型！）
# borderValue – （重点！）边界填充值; 默认情况下，它为0。
  return cv2.warpAffine(image, M, (nW, nH))#使用 OpenCV 库中的 warpAffine 函数对输入图像进行旋转操作，并返回旋转后的图像。

def decode(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)#获取灰色图像
    
    texts = barcodee(gray)#返回解码结果
    # print(texts)
    if texts == []:#解码结果为空
        print("未识别成功或没有识别对象")
        warning()#报错提醒
    else:
        for text in texts:
            tt = text.data.decode("utf-8")#字节对象转换成字符
            print("识别成功！识别结果为：" + tt)#打印识别结果

    barcodes = pyzbar.decode(gray)
    # 解码 返回了识别到的多个二维码对象
    if barcodes == None:#解码结果为空
        print("Error decoding barcodes")
    else:
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            print("识别成功！****** 识别结果是:" + barcodeData)
            #print(barcodeData)
        return image

def warning():#识别失败
  pass
def main_part():
    camera = cv2.VideoCapture(0)#开启摄像头
    interval = 0.2#等待时间
    while True:
        f, frame = camera.read()   
        if f == True:
            img = decode(frame)#调用自定义函数
            img = cv2.resize(img,(600,600),interpolation=cv2.INTER_CUBIC)#调用函数                 
            # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
            #src ：输入，原图像，即待改变大小的图像； 
            #输出图像的大小（600,600）
            # fx：width方向的缩放比例
            # fy：height方向的缩放比例
            # 输出，改变后的图像。这个图像和原图像具有相同的内容，只是大小和原图像不一样而已
            
            # interpolation 选项	所用的插值方法
            # INTER_NEAREST	最近邻插值
            # INTER_LINEAR	双线性插值（默认设置）
            # INTER_AREA	使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
            # INTER_CUBIC	4x4像素邻域的双三次插值
            # INTER_LANCZOS4	8x8像素邻域的Lanczos插值

            cv2.imshow('camera', img)#显示摄像头的信息
            # key = cv2.waitKey(1)#设置 waitKey(0) , 则表示程序会无限制的等待用户的按键事件。waitKey(1)等待一毫秒
            # if key == 27: #ESC(ASCII码为27)
            #     break
            if cv2.waitKey(1) == 27:
              break#youya
            sleep(interval)
        else: 
            print("Capture failed")#摄像头开启失败
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':#主函数入口
    main_part()