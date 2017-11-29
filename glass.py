import numpy as np
import cv2
from scipy import fftpack


#Load in images

glass_effect = cv2.imread("Images/glass_effect.jpg", cv2.IMREAD_COLOR)


glass_effect_hsl = cv2.cvtColor(glass_effect,cv2.COLOR_BGR2HLS)


def glass (image):

    image_hsl = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

    image_hsl[:,:,1]=glass_effect_hsl[:450,:800,1]

    glass_image = cv2.cvtColor(image_hsl,cv2.COLOR_HLS2BGR)
    return glass_image

#glass(image)


cv2.waitKey(0)
cv2.destroyAllWindows()