import numpy as np
import cv2
from scipy import fftpack


#Load in images
img = cv2.imread("Images/lemons.png", cv2.IMREAD_GRAYSCALE)
frosted = cv2.imread("Images/frostedlemons.png", cv2.IMREAD_GRAYSCALE)

def scale_dft_image(DFT_image):
    scaled = np.log(1. + np.real(np.absolute(DFT_image)))
    scaled = scaled / np.max(scaled) * 255.
    return np.uint8(np.round(scaled))

#Get the Fourier transform of both images
img_DFT = fftpack.fft2(img)
frosted_DFT = fftpack.fft2(frosted)

img_DFT_shift = fftpack.fftshift(img_DFT)
frosted_DFT_shift = fftpack.fftshift(frosted_DFT)

img_DFT_scaled = scale_dft_image(img_DFT_shift)
frosted_DFT_scaled = scale_dft_image(frosted_DFT_shift)

#Get the mask
mask = frosted_DFT_scaled - img_DFT_scaled

def glass (image,mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    img_hsl_float = np.float64(image)/255.
    transform = fftpack.fft2(img_hsl_float[:, :, 2])
    transform_shift = fftpack.fftshift(transform)
    transform_scaled = scale_dft_image(transform_shift)

    FTransform =  transform_scaled + mask
    Inverse = np.round(np.real(fftpack.ifft2(fftpack.ifftshift(FTransform))))
    return Inverse


cv2.imshow("window",img_DFT_scaled[::2,::2])
cv2.imshow("frost",frosted_DFT_scaled[::2,::2])
cv2.imshow("Mask", mask[::2,::2])

cv2.waitKey(0)
cv2.destroyAllWindows()