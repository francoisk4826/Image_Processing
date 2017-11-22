import numpy as np
import cv2

NOISE_AMOUNT = 5
MOTION_BLUR_SIZE = 35
BLUR_SIZE = 15


def lead(h, w):
    lead_base = np.zeros((h, w, 3), dtype=np.uint8)
    lead_base[:, :] = 100
    lead_base[:, 1 * (h // 5):2 * (h // 5)] = 180
    lead_base[:, 3 * (h // 5):4 * (h // 5)] = 180
    blur = np.ones((BLUR_SIZE, BLUR_SIZE)) / BLUR_SIZE**2
    lead_blur = cv2.filter2D(lead_base, -1, blur)
    for i in range(0, 50):
        lead_blur = cv2.filter2D(lead_blur, -1, blur)

    noise = np.zeros((h, w))
    cv2.randu(noise, 0, 100)
    lead_noise = np.copy(lead_blur)
    lead_noise[noise > 100 - NOISE_AMOUNT] = 255

    motion_blur = np.zeros((MOTION_BLUR_SIZE, MOTION_BLUR_SIZE))
    motion_blur[int((MOTION_BLUR_SIZE - 1) / 2), :] = np.ones(MOTION_BLUR_SIZE)
    motion_blur = motion_blur / MOTION_BLUR_SIZE
    lead_motion_blur = cv2.filter2D(lead_noise, -1, motion_blur)

    return lead_motion_blur
