import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

cur_direct = os.getcwd()
directory_path = '/images/'

files1 = [f for f in os.listdir(os.getcwd() + directory_path)]

for image in files1[:]:
    print(image)
    img = cv2.imread(''.join([cur_direct, directory_path, image]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([255, 20, 147])
    upper = np.array([255, 20, 150])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(hsv,hsv, mask=mask)
    edges = cv2.Canny(res, 50, 50, apertureSize=3)
    median = cv2.medianBlur(edges, 3)
    blur = cv2.GaussianBlur(edges, (3, 3), 0)

    edges = cv2.Canny(blur, 45, 50, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    k = np.shape(lines)
    #print(lines[:, 0])
    """T = []
    R = []
    RR = []
    TR = []
    RL = []
    TL = []
    TH = []
    RH = []

    if not k:
        pass
    else:
        for rho, theta in lines[:, 0]:
            #print(theta*180/np.pi)
            T.append(theta)
            R.append(rho)

            limdr = 0 * np.pi / 180
            limur = 9 * np.pi / 180

            if ((theta >= limdr) and (theta <= limur)):
                TR.append(theta)
                RR.append(rho)

            limdl = 175 * np.pi / 180
            limul = 185 * np.pi / 180

            if ((theta >= limdl) and (theta <= limul)):
                TL.append(theta)
                RL.append(rho)
            
            limhd = 95
            limhu = 85

            if (((theta >= 0) and (theta <= limhd))) or (((theta >= limhu) and (theta <= 180))):
                TH.append(theta)
                RH.append(rho)
            
        # -----------------------------------------------------------

        k = np.shape(img)
        if ((not TL) or (not RL)):

            theta2 = 0

        else:

            theta2 = np.mean(TL)
            rho2 = np.mean(RL)

            ar = np.cos(theta2)
            br = np.sin(theta2)
            x0r = ar * rho2
            y0r = br * rho2

            x1r = int(x0r + 5000 * (-br))
            y1r = int(y0r + 5000 * (ar))
            x2r = int(x0r - 5000 * (-br))
            y2r = int(y0r - 5000 * (ar))
            theta2a = theta2

            cv2.line(img, (int(x1r), int(y1r)), (x2r, y2r), (0, 0, 255), 3)

            # -------------------------RIGHT---------------------------------

        if ((not RR) or (not TR)):

            theta1 = 1

        else:

            theta1 = np.mean(TR)
            rho1 = np.mean(RR)

            al = np.cos(theta1)
            bl = np.sin(theta1)
            x0l = al * rho1
            y0l = bl * rho1

            x1l = int(x0l + 10000 * (-bl))
            y1l = int(y0l + 10000 * (al))
            x2l = int(x0l - 10000 * (-bl))
            y2l = int(y0l - 10000 * (al))

            theta1a = theta1

            cv2.line(img, (int(x1l), int(y1l)), (x2l, y2l), (255, 0, 0), 3)

            #edges = cv2.Canny(img, 20, 200)"""
    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222)
    plt.imshow(blur, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(res, cmap='gray')
    plt.title('Res'), plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    #plt.imshow(hsv)
    plt.imshow(hsv)

plt.show()