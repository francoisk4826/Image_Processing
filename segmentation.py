import numpy as np
import cv2

M = 10
SUPERPIXELS_TO_CHECK = 4


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def dist_color(image, s, x1, y1, x2, y2):
    l1 = image[x1, y1, 0]
    a1 = image[x1, y1, 1]
    b1 = image[x1, y1, 2]

    l2 = image[x2, y2, 0]
    a2 = image[x2, y2, 1]
    b2 = image[x2, y2, 2]

    dLab = np.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
    dxy = dist(x1, y1, x2, y2)
    d = dLab + (M/s * dxy)
    return d


def segment(img, k_r, k_c):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = np.shape(img)[:2]
    k = k_r * k_c
    s_r = h // k_r
    s_c = w // k_c
    s = np.sqrt(s_r * s_c)
    superpixel_centers = np.zeros((k_r, k_c, 3), dtype=np.int32)
    pixel_list = np.zeros((h, w), dtype=np.int32)
    n = 0

    for i in range(0, k_r):
        for j in range(0, k_c):
            superpixel_centers[i, j] = [s_r//2 + s_r*i, s_c//2 + s_c*j, n]
            n = n + 1

    for x in range(0, w):
        for y in range(0, h):
            superpixel_dist = np.zeros((k_r, k_c, 4))
            superpixel_semi_final = np.zeros((SUPERPIXELS_TO_CHECK, 4))
            distances = np.zeros((k))
            total_distances = np.zeros((SUPERPIXELS_TO_CHECK))
            n = 0

            for r in range(0, k_r):
                for c in range(0, k_c):
                    superpixel_dist[r, c][0] = superpixel_centers[r, c][0]
                    superpixel_dist[r, c][1] = superpixel_centers[r, c][1]
                    superpixel_dist[r, c][2] = superpixel_centers[r, c][2]
                    superpixel_dist[r, c][3] = dist(x, y, superpixel_dist[r, c][0], superpixel_dist[r, c][1])
                    distances[n] = superpixel_dist[r, c][3]
                    n = n + 1
            distances.sort()

            for i in range(0, SUPERPIXELS_TO_CHECK):
                for r in range(0, k_r):
                    for c in range(0, k_c):
                        if distances[i] == superpixel_dist[r, c][3]:
                            superpixel_semi_final[i] = superpixel_dist[r, c]

            for i in range(0, SUPERPIXELS_TO_CHECK):
                superpixel_semi_final[i][3] = dist_color(img, s, x, y, superpixel_semi_final[i][0], superpixel_semi_final[i][1])
                total_distances[i] = superpixel_semi_final[i][3]
            total_distances.sort()

            for i in range(0, SUPERPIXELS_TO_CHECK):
                if total_distances[0] == superpixel_semi_final[i][3]:
                    pixel_list[x, y] = superpixel_semi_final[i][2]

    return pixel_list
