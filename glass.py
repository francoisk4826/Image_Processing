import numpy as np
import cv2
from scipy import fftpack

GAMMA = 1.5
#Load in images

#display the mirror image
#glass_effect=glass_effect[:,::-1]


def normalize(image):
    max_intensities = np.max(image)
    min_intensities = np.min(image)
    return np.uint8(np.round(image-min_intensities)/(max_intensities-min_intensities)*255.0)


def glass(image):
    glass_image = cv2.imread("Images/watercolor.jpg", cv2.IMREAD_GRAYSCALE)

    image_height, image_width = np.shape(image)[:2]

    #this is here instead of hardcoding the height and width of the passed in superpixel image
    glass_effect=glass_image[:image_height,:image_width]

    for y in range(0, image_height):
        for x in range(0, image_width):
            image[y, x, 0] = np.uint8((np.uint64(image[y, x, 0]) + np.uint64(glass_effect[y, x])) // 2)
            image[y, x, 1] = np.uint8((np.uint64(image[y, x, 1]) + np.uint64(glass_effect[y, x])) // 2)
            image[y, x, 2] = np.uint8((np.uint64(image[y, x, 2]) + np.uint64(glass_effect[y, x])) // 2)


    normal_image = normalize(image)
    imgFloat = np.float64(normal_image) / 255.
    imgFloat = imgFloat ** GAMMA
    img = np.uint8(255. * (imgFloat))

    return img

#glass(image)


cv2.waitKey(0)
cv2.destroyAllWindows()