import imutils
import cv2
import numpy as np

portcamera = 0
cap = cv2.VideoCapture(portcamera)
cap.set(6,10)

lower = np.array([163, 168, 147])
upper = np.array([173, 226, 255])

while True :
    ret, img = cap.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_0 = cv2.inRange(gray, lower, upper)
    gray = cv2.bitwise_and(gray, gray, mask= mask_0)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 50, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

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
            cv2.drawContours(hsv, [screenCnt], -1, (0, 255, 0), 2)
            break

    # Display the resulting frame
    cv2.imshow('frame',hsv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break