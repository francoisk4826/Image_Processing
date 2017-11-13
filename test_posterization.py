import numpy as np
import cv2

img = cv2.imread("apple.jpg", cv2.IMREAD_COLOR)
# img = img[::4, ::4]
# img = img[::2, ::2]

# img[img < 86] = 42
# img[img < 172] = 130
# img[img >= 172] = 212

img[img < 64] = 32
img[img < 128] = 96
img[img < 192] = 170
img[img >= 192] = 224

# n = 2
# indices = np.arange(0, 256)
# divider = np.linspace(0, 255, n+1)[1]
# quantiz = np.int0(np.linspace(0, 255, n))
# color_levels = np.clip(np.int0(indices/divider), 0, n-1)
# palette = quantiz[color_levels]
# img2 = palette[img]
# img2 = cv2.convertScaleAbs(img2)

cv2.imshow("Test Posterization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()