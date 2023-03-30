import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np

def barcode(gray):
  texts = pyzbar.decode(gray)
  if texts == []:
    angle = barcode_angle(gray)
    if angle < -45:
      angle = -90 - angle
    texts = bar(gray, angle)
  if texts == []:
    gray = np.uint8(np.clip((1.1 * gray + 10), 0, 255))
    angle = barcode_angle(gray)
    #西瓜6写的，转载需声明
    if angle < -45:
      angle = -90 - angle
    texts = bar(gray, angle)
  return texts

def bar(image, angle):
  gray = image
  #西瓜6写的，转载需声明
  bar = rotate_bound(gray, 0 - angle)
  roi = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
  texts = pyzbar.decode(roi)
  return texts


def barcode_angle(image):
  gray = image
  #西瓜6写的，转载需声明
  ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
  kernel = np.ones((8, 8), np.uint8)
  dilation = cv2.dilate(binary, kernel, iterations=1)
  erosion = cv2.erode(dilation, kernel, iterations=1)
  erosion = cv2.erode(erosion, kernel, iterations=1)
  erosion = cv2.erode(erosion, kernel, iterations=1)
  
  contours, hierarchy = cv2.findContours(
    erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if len(contours) == 0:
    rect = [0, 0, 0]
  else:
    rect = cv2.minAreaRect(contours[0])
  return rect[2]

def rotate_bound(image, angle):
  (h, w) = image.shape[:2]
  (cX, cY) = (w // 2, h // 2)

  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])
  #西瓜6写的，转载需声明
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))

  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  return cv2.warpAffine(image, M, (nW, nH))

image=cv2.imread("image/img.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
texts = barcode(gray)
print(texts)
if texts==[]:
  print("未识别成功")
else:
  for text in texts:
    tt = text.data.decode("utf-8")
  print("识别成功")
  print(tt)

