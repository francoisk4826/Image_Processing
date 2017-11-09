import numpy as np
import cv2

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = cv2.imread("peppers_color.tif")
cv2.imshow("Maybe SLIC?", img)
img[::] = img[::] + 1
img = img_as_float(img)

print(np.shape(img))

segments_slic = slic(img, n_segments=150, sigma=4)
marked = mark_boundaries(img, segments_slic, color=0)
marked_h, marked_w = np.shape(marked)[:2]

print(np.shape(marked))

just_outlines = np.zeros((marked_h, marked_w, 3))
just_outlines[marked[::] != 0] = 255



cv2.imshow("Maybe SLIC?", marked)
cv2.waitKey(0)
cv2.destroyAllWindows()