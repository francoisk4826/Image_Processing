import numpy as np
import cv2
import math
from scipy import signal

THETA = 125
BLUR_SIZE = 3
GAMMA = 1.5


def normalize(image):
    max_intensities = np.max(image)
    min_intensities = np.min(image)
    return np.uint8(np.round(image-min_intensities)/(max_intensities-min_intensities)*255.0)


def lead(image_base):
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    image_sobel = signal.convolve2d(image_base, (math.cos(math.radians(THETA))*sobel_x + math.sin(math.radians(THETA))*sobel_y), mode='same')
    image_sobel = normalize(image_sobel)

    kernel = np.ones((BLUR_SIZE, BLUR_SIZE), np.float64) / BLUR_SIZE**2
    image_blur = cv2.filter2D(image_sobel, -1, kernel)

    img_float = np.float64(image_blur) / 255
    img_float = img_float ** GAMMA
    image_gamma = np.uint8(255 * img_float)

    return image_gamma


# cv2.imshow("Lead", lead(512, 512))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
