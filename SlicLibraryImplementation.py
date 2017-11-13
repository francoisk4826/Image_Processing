import numpy as np
import cv2
import random
from skimage.segmentation import slic

##Return a numpy array close to a circle for use as a kernel
def circle_n(radius):
    img = np.zeros((2*radius+1, 2*radius+1), dtype=np.uint8)
    cv2.circle(img, (radius, radius), radius, 1, -1)
    return img

## SLIC Parameters
NUM_SEGMENTS = 150 #100
SIGMA = 4
COMPACTNESS = 80 #50 #M parameter
ENFORCE_CONNECT = 1

##Import image
img = cv2.imread("Steven.jpg")
img[img != 255] = img[img != 255] + 1
print(np.shape(img))

##Set height and width of image
height, width = np.shape(img)[:2]

##Get SLIC Segments
segments_slic = slic(img, n_segments=NUM_SEGMENTS, sigma=SIGMA, enforce_connectivity=ENFORCE_CONNECT, compactness=COMPACTNESS)
print(np.max(segments_slic))
NUM_CREATED = np.max(segments_slic)

##Make array of 1-bit images for morphology
segments = np.zeros((NUM_CREATED+1,height,width), dtype=np.uint8)
for i in range(0, NUM_CREATED+1):
    segments[i][segments_slic == i] = 255

    ##Erode segments slightly
    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_ERODE, kernel= circle_n(2) )


##Set pixel intensities of each segment to the average of the segment
for i in range(0, NUM_CREATED+1):
    img[segments[i] == 255, 0] = np.average(img[segments[i] == 255, 0])
    img[segments[i] == 255, 1] = np.average(img[segments[i] == 255, 1])
    img[segments[i] == 255, 2] = np.average(img[segments[i] == 255, 2])


##Construct final image from 1-bit segments
finCanvas = np.zeros((height,width, 3), dtype = np.uint8)
for i in range(0, NUM_CREATED+1):
    finCanvas[segments[i] == 255] = img[segments[i] == 255]

##Make segment seperations gray
finCanvas[finCanvas == 0] = 80

##Shift some pixel intensities slightly to make the image not perfect
MAX_RAND = 35 #Max value a pixel intensity can change
for y in range(0, height):
    for x in range(0, width):
        if(random.randint(0, 5) == 5):
            rand_add = random.randint(5, MAX_RAND)

            if(finCanvas[y,x,0] + rand_add < 255):
               finCanvas[y,x,0] = finCanvas[y,x,0] + rand_add
            else:
                finCanvas[y, x, 0] = 255

            if (finCanvas[y, x, 1] + rand_add < 255):
                finCanvas[y, x, 1] = finCanvas[y, x, 1] + rand_add
            else:
                finCanvas[y, x, 1] = 255

            if (finCanvas[y, x, 2] + rand_add < 255):
                finCanvas[y, x, 2] = finCanvas[y, x, 2] + rand_add
            else:
                finCanvas[y, x, 2] = 255

            #if((finCanvas[y,x,0] + rand_add < 255) & (finCanvas[y,x,1] + rand_add < 255) & (finCanvas[y,x,2] + rand_add < 255)):
            #  print("yes", finCanvas[y,x])
            #   finCanvas[y,x] = finCanvas[y,x] + rand_add#random.randint(5,MAX_RAND
            #else:
            #    finCanvas[y,x] = 255


##Cut off pixel intensities too large or small
finCanvas[finCanvas < 0] = 0
finCanvas[finCanvas > 255] = 255

##Display image
cv2.imshow("Literally The Best Thing You've Ever Seen In Your Life", finCanvas)
cv2.waitKey(0)
cv2.destroyAllWindows()