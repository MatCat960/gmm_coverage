#!/usr/bin/env python3

import math
import time
import numpy as np
from sklearn.mixture import GaussianMixture
from roipoly import RoiPoly
from matplotlib import pyplot as plt

AREA_X = 40
AREA_Y = 20
COMPONENTS_NUM = 12

# function to check if point is inside polygon
def isInside(x, y, xp, yp):
    c = False
    j = len(xp)-1
    for i in range(len(xp)):
        if (((yp[i] > y) != (yp[j] > y)) and (x < (xp[j]-xp[i]) * (y-yp[i]) / (yp[j]-yp[i]) + xp[i])):
            c = not c
        j = i
    return c



def main():
    image = plt.imread('blank.jpg')
    plt.xlim([0,AREA_X])
    plt.ylim([0,AREA_Y])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    # get ROI
    roi = RoiPoly(color='r')
    coords = roi.get_roi_coordinates()
    xr = []             # x coordinates of ROI
    yr = []             # y coordinates of ROI
    # coords = coords[0]
    for tuple in coords:
        xr.append(tuple[0])
        yr.append(tuple[1])

    # get min and max value of x and y
    xmin = min(xr)
    xmax = max(xr)
    ymin = min(yr)
    ymax = max(yr)

    print("x range: {} - {}".format(xmin, xmax))
    print("y range: {} - {}".format(ymin, ymax))

    # generate 2000 points in ROI
    xp = []
    yp = []
    for i in range(4000):
        xt = xmin + np.random.random()*(xmax-xmin)
        yt = ymin + np.random.random()*(ymax-ymin)
        if isInside(xt, yt, xr, yr):
            xp.append(xt)
            yp.append(yt)

    print("Number of points: {}".format(len(xp)))

    GMModel = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
    GMModel.fit(np.column_stack((xp, yp)))
    # calculate BIC
    # bic = GMModel.bic(np.column_stack((xp, yp)))

    # get means and covariances
    means = GMModel.means_
    covariances = GMModel.covariances_
    mix = GMModel.weights_
    print("Means: {}".format(means))
    print("Coveriances: {}".format(covariances))
    print("Mixture proportions: {}".format(mix))


    # generate grid
    # xgrid = np.linspace(0,image.shape[0])
    # ygrid = np.linspace(0,image.shape[1])
    # X, Y = np.meshgrid(xgrid, ygrid)
    plt.scatter(xp[:], yp[:], marker='.', color='k', s=1)
    plt.title("Points inside ROI")
    plt.xticks([])
    plt.yticks([])
    plt.show()



    # img = plt.imread()
    # img.plot(xp, yp, 'ro')
    # plt.imshow(img)



main()



    


