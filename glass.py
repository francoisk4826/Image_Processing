import numpy as np
import cv2
from scipy import fftpack


#Load in images

glass_effect = cv2.imread("Images/my_glasseffect1.png", cv2.IMREAD_COLOR)

#display the mirror image
#glass_effect=glass_effect[:,::-1]


glass_effect_hsl = cv2.cvtColor(glass_effect,cv2.COLOR_BGR2HLS)


def glass (image):
    image_height, image_width = np.shape(image)[:2]
    image_hsl = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

    image_hsl[:,:,1]=glass_effect_hsl[:image_height,:image_width,1]

    glass_image = cv2.cvtColor(image_hsl,cv2.COLOR_HLS2BGR)
    return glass_image

#glass(image)


cv2.waitKey(0)
cv2.destroyAllWindows()