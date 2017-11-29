import numpy as np
import cv2
import segmentation
import lead
import glass
from scipy import fftpack

##Return a numpy array close to a circle for use as a kernel
def circle_n(radius):
    img = np.zeros((2*radius+1, 2*radius+1), dtype=np.uint8)
    cv2.circle(img, (radius, radius), radius, 1, -1)
    return img

plus = np.array([[0,1,0],[1,1,1],[0,1,0]])

##Import image
img = cv2.imread("Images/Steven.jpg")
img[img != 255] = img[img != 255] + 1
print(np.shape(img))

##Set height and width of image
height, width = np.shape(img)[:2]

##Get SLIC Segments
segments_slic = segmentation.segment(img, 10, 10, 250, 1)
print(np.max(segments_slic))
NUM_CREATED = np.max(segments_slic)

##Make array of 1-bit images for morphology
segments = np.zeros((NUM_CREATED+1, height, width), dtype=np.uint8)
for i in range(0, NUM_CREATED+1):
    segments[i][segments_slic == i] = 255
    #cv2.imshow(str(i), segments[i])

    i2 = str(i) + str(2)
    ##Morphology things and stuff
    segments[i] = cv2.medianBlur(segments[i], 21)
    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_CLOSE, kernel=circle_n(2))
    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_OPEN, kernel=circle_n(2))
    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_ERODE, kernel=circle_n(2))
    #cv2.imshow(i2, segments[i])

##Set pixel intensities of each segment to the average of the segment
for i in range(0, NUM_CREATED+1):
    img[segments[i] == 255, 0] = np.average(img[segments[i] == 255, 0])
    img[segments[i] == 255, 1] = np.average(img[segments[i] == 255, 1])
    img[segments[i] == 255, 2] = np.average(img[segments[i] == 255, 2])

img = glass.glass(img)


##Construct final image from 1-bit segments
finCanvas = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(0, NUM_CREATED+1):
    finCanvas[segments[i] == 255] = img[segments[i] == 255]

##Make segment seperations gray
lead_img = lead.lead(height, width)
##This needs to be done better
for y in range(0, height):
    for x in range(0, width):
        if finCanvas[y, x].all() == 0:
            finCanvas[y, x] = lead_img[y, x]

##Cut off pixel intensities too large or small
finCanvas[finCanvas < 0] = 0
finCanvas[finCanvas > 255] = 255

##Display image
cv2.imshow("Literally The Best Thing You've Ever Seen In Your Life", finCanvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
