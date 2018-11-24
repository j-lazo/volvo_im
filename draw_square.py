import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# load the image
cur_direct = os.getcwd()
directory_path = '/images/pink_sqr/'
files1 = [f for f in os.listdir(os.getcwd() + directory_path)]

lower = np.array([163, 168, 147])
upper = np.array([173, 226, 255])

for file in files1[:]:
    img = cv2.imread(''.join([cur_direct, directory_path, file]))

    lower = np.array([163,168,147])
    upper = np.array([173,226,255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_0 = cv2.inRange(gray, lower, upper)
    gray = cv2.bitwise_and(gray, gray, mask= mask_0)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 50,apertureSize = 3)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)

    ratio = img.shape[0] / 500.0

    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(approx)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(gray, [screenCnt], -1, (0, 255, 0), 2)
    plt.figure()
    plt.subplot(121)
    plt.imshow(gray)
    plt.subplot(122)
    plt.imshow(hsv)
    plt.show()
