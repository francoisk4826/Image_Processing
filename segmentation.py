import numpy as np
import cv2


def segment(img, k_r, k_c, m, num_iterations):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = np.shape(img)[:2]
    k = k_r * k_c
    s_r = h // k_r
    s_c = w // k_c
    s = np.sqrt(s_r * s_c)
    superpixel_centers_xy = np.zeros((k, 2), dtype=np.int32)
    superpixel_centers_lab = np.zeros((k, 3), dtype=np.int32)
    coordinates = np.zeros((h, w, 2), dtype=np.int32)
    distances = np.zeros((k, h, w), dtype=np.float64)
    pixel_list = np.zeros((h, w), dtype=np.int32)

    for y in range(0, h):
        for x in range(0, w):
            coordinates[y, x] = np.array([y, x])

    for n in range(0, num_iterations):
        ##Setting initial superpixel centers
        if n == 0:
            for i in range(0, k_r):
                for j in range(0, k_c):
                    superpixel_centers_xy[(i * k_c) + j] = [s_r // 2 + s_r * i, s_c // 2 + s_c * j]
                    superpixel_centers_lab[(i * k_c) + j] = img[superpixel_centers_xy[(i * k_c) + j][0], superpixel_centers_xy[(i * k_c) + j][1]]

        ##SLIC
        print("Iteration #", n+1)
        for r in range(0, k_r):
            for c in range(0, k_c):
                d_xy = np.sqrt(np.sum((superpixel_centers_xy[(r * k_c) + c] - coordinates) ** 2., axis=2))
                d_lab = np.sqrt(np.sum((superpixel_centers_lab[(r * k_c) + c] - img) ** 2., axis=2))
                d = d_lab + (m / s * d_xy)
                distances[(r * k_c) + c] = d

        ##Assigning superpixel to each pixel
        pixel_list = np.argmin(distances, axis=0)

        ##Recalculating superpixel centers
        print("Recalculating centers")
        if n < num_iterations-1:
            for m in range(0, k):
                row, col = np.where(pixel_list == m)
                superpixel_centers_xy[m] = [np.average(row), np.average(col)]

            for a in range(0, k_r):
                for b in range(0, k_c):
                    superpixel_centers_lab[(a * k_c) + b] = img[superpixel_centers_xy[(a * k_c) + b][0], superpixel_centers_xy[(a * k_c) + b][1]]

    return pixel_list

