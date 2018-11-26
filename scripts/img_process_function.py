import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt



def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    height, width, chan = np.shape(image)

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    #dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    dst = np.array([[500, 1000], [1000, 1000], [1000, 1500], [500, 1500]], dtype='float32')

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (2*np.shape(image)[0], int(1.5*np.shape(image)[1])))
    #warped = cv2.warpPerspective(image, M, (1000, 1000))
    # return the warped image
    return warped


def find_square(img):

    # limits for the mask
    lower = np.array([163, 120, 130])
    upper = np.array([173, 226, 255])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_0 = cv2.inRange(gray, lower, upper)
    gray = cv2.bitwise_and(gray, gray, mask= mask_0)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 50,apertureSize = 3)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)

    ratio = img.shape[0] / 1

    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(gray, [screenCnt], -1, (0, 255, 0), 2)
            #print(screenCnt)
            break
        else:
            #print(len(approx))
            screenCnt = np.zeros(np.array([2, 4]))

    if cnts == []:
        screenCnt = np.zeros(np.array([2, 4]))

    return screenCnt, gray


def roller_calc(img):

    # limits for the mask
    lower = np.array([163, 120, 130])
    upper = np.array([173, 226, 255])

    #lower = np.array([160, 80, 130])
    #upper = np.array([164, 210, 255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_0 = cv2.inRange(gray, lower, upper)
    gray = cv2.bitwise_and(gray, gray, mask= mask_0)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 50, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    ratio = img.shape[0] / 1

    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen

        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(gray, [screenCnt], -1, (0, 255, 0), 2)
            warped = four_point_transform(img, screenCnt.reshape(4, 2))

            break
        else:
            #print(len(approx))
            warped = np.zeros(np.shape(img))

    """plt.figure()
    plt.subplot(131)
    plt.imshow(hsv)
    plt.subplot(132)
    plt.imshow(warped)
    plt.subplot(133)
    plt.imshow(gray)
    plt.show()"""

    return warped


def look_for_red_points(ima):

    lower_lim = np.array([0, 0, 245])
    upper_lim = np.array([10, 10, 255])

    mask_red = cv2.inRange(ima, lower_lim, upper_lim)
    redish = cv2.bitwise_and(ima, ima, mask= mask_red)

    gray = cv2.cvtColor(redish, cv2.COLOR_BGR2HSV)
    ret, th1 = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY)

    non_zero_points = np.where(th1[:, :, 2] > 0)
    mean_x = np.mean(non_zero_points[0])
    xpoints_std = np.std(non_zero_points[0])
    mean_y = np.mean(non_zero_points[1])
    ypoints_std = np.std(non_zero_points[1])

    frecuencies_y = (np.histogram(non_zero_points[0])[0])
    lines_y = (np.where(frecuencies_y > 0))

    if lines_y[0] != []:
        line_y1 = np.histogram(non_zero_points[0])[1][lines_y[0][0]]
    else:
        line_y1 = 0

    if lines_y[0] != []:
        line_y2 = np.histogram(non_zero_points[0])[1][lines_y[0][1]]
    else:
        line_y2 = 0

    #print(line_y1)
    #print(line_y2)
    #plt.hist(non_zero_points[0])

    #plt.figure()
    #print(np.histogram(non_zero_points[1]))
    #plt.hist(non_zero_points[1])

    return line_y1, line_y2, 0, 0


def calculate_distances(points_1, points_2, rate, curvature=False):

    if points_1 == points_2:
        result = 'Error, not possible to calculate'
    else:

        if curvature is True:
            parameter = 2/1.5
        else:
            parameter = 1

        result = abs(points_1 - points_2)*rate*parameter

    return result


def magic(img, curvature_calc):

    plots = False

    reference = np.zeros(np.array([2, 4]))
    new_perspective = roller_calc(img)
    points, image_squares = find_square(new_perspective)
    points = order_points(points.reshape(4, 2))
    #plt.figure()
    #plt.imshow(new_perspective)
    #plt.show()
    if not (np.array_equal(points, reference)):
        ratio_1 = abs(40 / (points[0][0] - points[1][0]))
        ratio_2 = abs(40 / (points[0][1] - points[3][1]))
        ratio_3 = abs(40 / (points[1][1] - points[2][1]))
        ratio_4 = abs(40 / (points[2][0] - points[3][0]))
        # print(ratio_1, ratio_2, ratio_3, ratio_4)
        mean_rate = np.mean([ratio_1, ratio_2, ratio_3, ratio_4])
        deviation_rate = np.std([ratio_1, ratio_2, ratio_3, ratio_4])

    pt_1, pt_2, std_x, std_y = look_for_red_points(new_perspective)

    if not (np.isnan(mean_rate)):
        magic_number = (calculate_distances(pt_1, pt_2, mean_rate, curvature_calc))

    else:
        'error, not possible to measure, take another picture or change the points'

    if plots is True:

        pass
        plt.figure()
        plt.subplot(121)
        plt.imshow(new_perspective)
        plt.subplot(122)
        plt.imshow(image_squares)
        plt.show()

    return magic_number


