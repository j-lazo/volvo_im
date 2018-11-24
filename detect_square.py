# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())

# load the image

cur_direct = os.getcwd()
directory_path = '/images/pink_sqr/'
files1 = [f for f in os.listdir(os.getcwd() + directory_path)]

# define the list of boundaries
boundaries = [([163, 168, 147], [173, 218, 255])
              ]



for image in files1[:]:
    print(image)
    image = cv2.imread(''.join([cur_direct, directory_path, image]))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    Y = np.arange(hsv.shape[0])
    X = np.arange(hsv.shape[1])
    X, Y = np.meshgrid(X, Y)

    # loop over the boundaries
    for i, (lower, upper) in enumerate(boundaries[:]):
        # create NumPy arrays from the boundaries
        lower = np.array(lower)
        upper = np.array(upper)

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(hsv, hsv, mask=mask)
        edges = cv2.Canny(output, 45, 50, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        # show the images
        plt.figure()
        plt.title('Filter'+str(i))
        plt.subplot(121)
        plt.imshow(edges)
        plt.subplot(122)
        plt.imshow(hsv)
        plt.show()