import numpy as np
import cv2

SUPERPIXELS_TO_CHECK = 4


def dist(image, superpixel):
    return np.sqrt(np.sum((image-superpixel)**2., axis=1))


def dist_color(image, superpixel):
    return np.sqrt(np.sum((image-superpixel)**2., axis=1))


def segment(img, k_r, k_c, m):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = np.shape(img)[:2]
    k = k_r * k_c
    s_r = h // k_r
    s_c = w // k_c
    s = np.sqrt(s_r * s_c)
    superpixel_centers_xy = np.zeros((k, 2), dtype=np.int32)
    superpixel_centers_lab = np.zeros((k, 3), dtype=np.int32)
    pixel_list = np.zeros((h, w), dtype=np.int32)

    for i in range(0, k_r):
        for j in range(0, k_c):
            superpixel_centers_xy[(i * k_c) + j] = [s_r // 2 + s_r * i, s_c // 2 + s_c * j]
            superpixel_centers_lab[(i * k_c) + j] = img[superpixel_centers_xy[(i * k_c) + j][0], superpixel_centers_xy[(i * k_c) + j][1]]

    for x in range(0, w):
        for y in range(0, h):
            d_xy = dist(np.array([x, y]), superpixel_centers_xy)
            d_lab = dist_color(img[x, y], superpixel_centers_lab)
            d = d_lab + (m / s * d_xy)
            superpixel_num = np.argmin(d)
            pixel_list[x, y] = superpixel_num

            if y == 0:
                print(int((x/h)*100), "%")

    return pixel_list
