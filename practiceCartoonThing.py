import numpy as np
import cv2
import filters
import math


kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
kernel2 = np.ones((3,3))

img_color = cv2.imread("peppers_color.tif", cv2.IMREAD_COLOR)

img = cv2.imread("peppers_gray.tif", cv2.IMREAD_GRAYSCALE)
blurred = cv2.medianBlur(img, 3)
blurred = filters.bad_gaussian_blur(img)
for i in range(0, 3):
   blurred = filters.bad_gaussian_blur(blurred)
img_1bit = cv2.Canny(blurred, 10, 100)

#img_1bit = cv2.morphologyEx(img_1bit, cv2.MORPH_DILATE, kernel)
#img_1bit = cv2.morphologyEx(img_1bit, cv2.MORPH_DILATE, kernel)
img_1bit = cv2.morphologyEx(img_1bit, cv2.MORPH_DILATE, kernel2)

img_1bit = np.invert(img_1bit)
img_color = cv2.medianBlur(img_color, 11)
img_color[img_1bit == 0] = 5


cv2.imshow("Probably Something", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()