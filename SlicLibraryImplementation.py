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
img = cv2.imread("Images/dog.jpg")
img[img != 255] = img[img != 255] + 1
print(np.shape(img))

##Set height and width of image
height, width = np.shape(img)[:2]

#Set size of median blur based on size of image
median_size = np.int64(np.ceil(height*width*.00008))
if(median_size % 2 == 0):
    median_size = median_size + 1
if(median_size > 15):
    median_size = 15

print("Median size:", median_size)

##Get SLIC Segments
#(img, rows, columns, m, iterations)
segments_slic = segmentation.segment(img, 12,12, 100, 10)
print(np.max(segments_slic))
NUM_CREATED = np.max(segments_slic)

superpixel_centers_xy = np.zeros((100,2))
# distances = np.zeros((100, height, width), dtype=np.float64)
#
for m in range(0, 100):
  row, col = np.where(segments_slic == m)
  superpixel_centers_xy[m] = [np.average(row), np.average(col)]

# for r in range(0, 10):
#     for c in range(0, 10):
#         d = np.sqrt(np.sum((superpixel_centers_xy[(r * 10) + c] - superpixel_centers_xy) ** 2., axis=2))
#         #d_lab = np.sqrt(np.sum((superpixel_centers_lab[(r * k_c) + c] - img) ** 2., axis=2))
#         #d = d_lab + (m / s * d_xy)
#         distances[(r * 10) + c] = d


#Combine like segments:
# for i in range(NUM_CREATED+1):
#     print(np.size(segments_slic[segments_slic == i]))
#     if(np.size(segments_slic[segments_slic == i]) < 1000):
#         print("DOING THINGS")
#         segments_slic[segments_slic == i] = i+1


##Make array of 1-bit images for morphology
segments = np.zeros((NUM_CREATED+1, height, width), dtype=np.uint8)
for i in range(0, NUM_CREATED+1):
    segments[i][segments_slic == i] = 255
    #cv2.imshow(str(i), segments[i])

    i2 = str(i) + str(2)
    ##Morphology things and stuff
    segments[i] = cv2.medianBlur(segments[i], median_size)
    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_CLOSE, kernel=circle_n(2))
    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_OPEN, kernel=circle_n(6))

    img[segments[i] == 255, 0] = np.average(img[segments[i] == 255, 0])
    img[segments[i] == 255, 1] = np.average(img[segments[i] == 255, 1])
    img[segments[i] == 255, 2] = np.average(img[segments[i] == 255, 2])

    segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_ERODE, kernel=circle_n(2))

#Set pixel intensities of each segment to the average of the segment
for i in range(0, NUM_CREATED+1):
    img[segments[i] == 255, 0] = np.average(img[segments[i] == 255, 0])
    img[segments[i] == 255, 1] = np.average(img[segments[i] == 255, 1])
    img[segments[i] == 255, 2] = np.average(img[segments[i] == 255, 2])


img = glass.glass(img)


##Construct final image from 1-bit segments
finCanvas = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(0, NUM_CREATED+1):
    finCanvas[segments[i] == 255] = img[segments[i] == 255]
#     y,x = np.round(superpixel_centers_xy[i])
#     y = np.int64(y)
#     x = np.int64(x)
#     #print(y,x)
#     finCanvas[y-1: y+1,x-1:x+1,0] = 255
#     finCanvas[y-1: y+1,x-1:x+1,1] = 255
#     finCanvas[y-1: y+1,x-1:x+1,2] = 255

#Make segment seperations gray
lead_base = np.zeros((height, width, 3), dtype=np.uint8)
lead_base[finCanvas == 0] = 20
lead_base = cv2.cvtColor(lead_base, cv2.COLOR_BGR2GRAY)
lead_img = lead.lead(lead_base)
for y in range(0, height):
    for x in range(0, width):
        if finCanvas[y, x].all() == 0:
            finCanvas[y, x] = lead_img[y, x]

##Cut off pixel intensities too large or small
finCanvas[finCanvas < 0] = 0
finCanvas[finCanvas > 255] = 255

##Display image
print("All done :)")
cv2.imshow("Literally The Best Thing You've Ever Seen In Your Life", finCanvas)
cv2.imwrite("peps.png", finCanvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
