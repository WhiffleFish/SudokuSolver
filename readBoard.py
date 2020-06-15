import cv2 as cv
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


def isolate_board(img):
    edges = canny(img, sigma=1).astype(np.uint8)

    test_angles = np.linspace(-np.pi/2, np.pi/2, 360)  # Sweep -90° to +90°
    h, theta, d = hough_line(edges, theta=test_angles)

    top = img.shape[0] + 1
    bottom = -1
    left = img.shape[1] + 1
    right = -1

    # Horizontal Line: theta ~= pi/2
    # Vertical Line: theta ~= 0
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        # If line is Vertical
        if -45 < np.rad2deg(angle) < 45:
            # If line is furthest Left
            if dist < left: left = dist
            # If line is furthest Right
            if dist > right: right = dist
        else:  # Line is Horizontal
            # If line is furthest Down
            if dist > bottom: bottom = dist
            # If line is furthest Up
            if dist < top: top = dist

    return img[int(top):int(bottom), int(left):int(right)]
