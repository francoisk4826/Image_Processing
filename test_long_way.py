import numpy as np
import cv2
from PIL import Image

# img_start = Image.open("nebula-wallpaper-hd.jpg")
# img_start = Image.open("Green Cavern.jpg")
img_start = cv2.imread("peppers_color.tif")

img_norm = cv2.cvtColor(img_start, cv2.COLOR_BGR2RGB)
img_array = Image.frombytes(mode='RGB', size=np.shape(img_norm)[:2], data=img_norm)

img_rPalette = Image.Image.convert(self=img_array, mode='P', palette=Image.ADAPTIVE, colors=15)

Image.Image.show(img_array)
Image.Image.show(img_rPalette)
