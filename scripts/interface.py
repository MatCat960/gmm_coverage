#!/usr/bin/env python3

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from roipoly import RoiPoly
from matplotlib import pyplot as plt

# ROS imports
import rclpy
from rclpy.node import Node
from gmm_msgs.msg import GMM, Gaussian
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_prefix


AREA_X = 3.0
AREA_Y = 3.0
COMPONENTS_NUM = 4

# function to check if point is inside polygon
def isInside(x, y, xp, yp):
    c = False
    j = len(xp)-1
    for i in range(len(xp)):
        if (((yp[i] > y) != (yp[j] > y)) and (x < (xp[j]-xp[i]) * (y-yp[i]) / (yp[j]-yp[i]) + xp[i])):
            c = not c
        j = i
    return c





class HSInterface(Node):
    def __init__(self):
        super().__init__('hs_interface')
        self.publisher_ = self.create_publisher(GMM, '/gaussian_mixture_model', 1)
        timer_period = 1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.gmm_msg = GMM()
        self.draw_roi()


    def draw_roi(self):
        # find blank image
        pkg_path = get_package_prefix('gmm_coverage')
        pkg_path = os.path.join(pkg_path, '..', '..', 'src', 'gmm_coverage')
        path = os.path.join(pkg_path, "scripts/blank.jpg")

        # show graphical interface
        image = plt.imread(path)
        plt.xlim([0,AREA_X])
        plt.ylim([0,AREA_Y])
        plt.xticks([])
        plt.yticks([])

        # num = input("Enter number of components: ")
        # COMPONENTS_NUM = int(num)

        plt.imshow(image)

        # get ROI
        roi = RoiPoly(color='r')
        coords = roi.get_roi_coordinates()
        xr = []             # x coordinates of ROI
        yr = []             # y coordinates of ROI
        for tuple in coords:
            xr.append(tuple[0])
            yr.append(tuple[1])

        # get min and max value of x and y
        xmin = min(xr)
        xmax = max(xr)
        ymin = min(yr)
        ymax = max(yr)

        # generate 4000 points in ROI
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


        # Re-elaborate mean points to fit environment with origin in center
        for m in means:
            m[0] -= 0.5*AREA_X
            m[1] -= 0.5*AREA_Y

        print("Means: {}".format(means))
        print("Coveriances: {}".format(covariances))
        print("Mixture proportions: {}".format(mix))

        for i in range(len(means)):
            g = Gaussian()
            mean_pt = Point()

            mean_pt.x = means[i][0]
            mean_pt.y = means[i][1]
            mean_pt.z = 0.0
            g.mean_point = mean_pt
            for j in range(len(covariances[i])):
                g.covariance.append(covariances[i][j][0])
                g.covariance.append(covariances[i][j][1])

            self.gmm_msg.gaussians.append(g)
            self.gmm_msg.weights.append(mix[i])

        # show points in roi
        # plt.scatter(xp[:], yp[:], marker='.', color='k', s=1)
        # plt.title("Points inside ROI")
        # plt.xticks([])
        # plt.yticks([])
        # plt.show(block=False)
        


    def timer_callback(self):
        self.publisher_.publish(self.gmm_msg)





def main(args=None):
    rclpy.init(args=args)
    hsiObj = HSInterface()    
    rclpy.spin(hsiObj)

    hsiObj.destroy_node()
    rclpy.shutdown()
    



if __name__ == '__main__':
    main()



    


