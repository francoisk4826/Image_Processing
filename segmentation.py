import numpy as np
import cv2

SUPERPIXELS_TO_CHECK = 10


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

    print("centers:", np.shape(superpixel_centers_xy))

    print("Initial SLIC")
    for y in range(0, h):
        for x in range(0, w):
            d_xy = dist(np.array([y, x]), superpixel_centers_xy)
            d_lab = dist_color(img[y, x], superpixel_centers_lab)
            d = d_lab + (m / s * d_xy)
            superpixel_num = np.argmin(d)
            pixel_list[y, x] = superpixel_num


    for i in range(0, 10):
      print("Recenter and Slic #", i+1)

      print("RE-CENTERING")
      for i in range(0, k):
          row, col = np.where(pixel_list == i)
          superpixel_centers_xy[i] = [np.average(row), np.average(col)]

      for i in range(0, k_r):
        for j in range(0, k_c):
          superpixel_centers_lab[(i * k_c) + j] = img[superpixel_centers_xy[(i * k_c) + j][0], superpixel_centers_xy[(i * k_c) + j][1]]

      print("RE-SLICCING")
      for y in range(0, h):
         for x in range(0, w):
            d_xy = dist(np.array([y, x]), superpixel_centers_xy)
            d_lab = dist_color(img[y, x], superpixel_centers_lab)
            d = d_lab + (m / s * d_xy)
            superpixel_num = np.argmin(d)
            pixel_list[y, x] = superpixel_num

    return pixel_list

