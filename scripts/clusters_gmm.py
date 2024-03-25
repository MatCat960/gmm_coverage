#!/usr/bin/env python3

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

# ROS imports
import rospy
from gmm_msgs.msg import GMM, Gaussian
from geometry_msgs.msg import Point

AREA_W = 20.0
COMPONENTS_NUM = 2
SAVE_PTS = True
RANDOM_GMM = False
DRAW_GMM = True

# function to check if point is inside polygon
def isInside(x, y, xp, yp):
    c = False
    j = len(xp)-1
    for i in range(len(xp)):
        if (((yp[i] > y) != (yp[j] > y)) and (x < (xp[j]-xp[i]) * (y-yp[i]) / (yp[j]-yp[i]) + xp[i])):
            c = not c
        j = i
    return c



def gauss_pdf(x, y, mean, covariance):
  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob


def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  s = len(means)
  for i in range(s):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])

  return prob

class ClustersGMM():
    def __init__(self):
        self.publisher_ = rospy.Publisher("/gaussian_mixture_model", GMM, queue_size=1)
        timer_period = 1.0
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.timer_callback)
        self.gmm_msg = GMM()
        cov = [0.5, 0.0, 0.0, 0.5]
        w = 1.0/COMPONENTS_NUM
        means = np.zeros((COMPONENTS_NUM, 2))
        
        if RANDOM_GMM:
            means = AREA_W*np.random.rand(COMPONENTS_NUM, 2)
        else:
            # means = np.array([[5.0, 5.0], [5.0, 15.0], [15.0, 5.0], [15.0, 15.0]])
            means = np.array([[5.0, 5.0], [5.0, 15.0]])
            

        for i in range(COMPONENTS_NUM):
            g = Gaussian()
            mean_pt = Point()
            mean_pt.x = means[i,0]
            mean_pt.y = means[i,1]
            mean_pt.z = 0.0
            g.mean_point = mean_pt
            g.covariance = cov
            self.gmm_msg.gaussians.append(g)
            self.gmm_msg.weights.append(w)

        print("GMM means: {}".format(means))

        if DRAW_GMM:
            discretization_pts = 200
            np_cov = np.array(cov).reshape(2,2)
            np_covs = [np_cov for _ in range(COMPONENTS_NUM)]
            xg = np.linspace(0.0, AREA_W, discretization_pts)
            yg = np.linspace(0.0, AREA_W, discretization_pts)
            Xg, Yg = np.meshgrid(xg, yg)
            z = gmm_pdf(Xg, Yg, means, np_covs, self.gmm_msg.weights)
            z = z.reshape(discretization_pts, discretization_pts)
            print("Z shape: ", z.shape)
            z_max = np.abs(z).max()
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            c = ax.pcolormesh(Xg, Yg, z, cmap='RdGy_r', vmin=-z_max, vmax=z_max)
            plt.xticks([]); plt.yticks([])
            plt.axis('equal')
            plt.show()


            


    # def draw_roi(self):
    #     # find blank image
    #     rospack = rospkg.RosPack()
    #     # rospack.list()
        
    #     pkg_path = rospack.get_path('gmm_coverage')
    #     print("Package path: {}".format(pkg_path))
    #     pkg_path = os.path.join(pkg_path, '..', '..', 'src', 'gmm_coverage')
    #     path = os.path.join(pkg_path, "scripts/blank.jpg")

    #     # show graphical interface
    #     image = plt.imread(path)
    #     plt.xlim([0,AREA_X])
    #     plt.ylim([0,AREA_Y])
    #     plt.xticks([])
    #     plt.yticks([])

    #     # num = input("Enter number of components: ")
    #     # COMPONENTS_NUM = int(num)

    #     plt.imshow(image)

    #     # get ROI
    #     roi = RoiPoly(color='r')
    #     coords = roi.get_roi_coordinates()
    #     xr = []             # x coordinates of ROI
    #     yr = []             # y coordinates of ROI
    #     for tuple in coords:
    #         xr.append(tuple[0])
    #         yr.append(tuple[1])
    #         if SAVE_PTS:
    #             with open('roi_pts.txt', 'a') as f:
    #                 f.write(str(tuple[0]) + " " + str(tuple[1]) + "\n")

    #     # get min and max value of x and y
    #     xmin = min(xr)
    #     xmax = max(xr)
    #     ymin = min(yr)
    #     ymax = max(yr)

    #     # generate 4000 points in ROI
    #     xp = []
    #     yp = []
    #     i = 0
    #     while i < 4000:
    #         xt = xmin + np.random.random()*(xmax-xmin)
    #         yt = ymin + np.random.random()*(ymax-ymin)
    #         if isInside(xt, yt, xr, yr):
    #             xp.append(xt)
    #             yp.append(yt)
    #             i += 1

    #     print("Number of points: {}".format(len(xp)))

    #     GMModel = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
    #     GMModel.fit(np.column_stack((xp, yp)))
        
    #     # calculate BIC
    #     # bic = GMModel.bic(np.column_stack((xp, yp)))

    #     # get means and covariances
    #     means = GMModel.means_
    #     covariances = GMModel.covariances_
    #     mix = GMModel.weights_


    #     # Re-elaborate mean points to fit environment with origin in center
    #     # for m in means:
    #     #     m[0] -= 0.5*AREA_X
    #     #     m[1] -= 0.5*AREA_Y

    #     print("Means: {}".format(means))
    #     print("Coveriances: {}".format(covariances))
    #     print("Mixture proportions: {}".format(mix))

    #     for i in range(len(means)):
    #         g = Gaussian()
    #         mean_pt = Point()

    #         mean_pt.x = means[i][0]
    #         mean_pt.y = means[i][1]
    #         mean_pt.z = 0.0
    #         g.mean_point = mean_pt
    #         for j in range(len(covariances[i])):
    #             g.covariance.append(covariances[i][j][0])
    #             g.covariance.append(covariances[i][j][1])

    #         self.gmm_msg.gaussians.append(g)
    #         self.gmm_msg.weights.append(mix[i])

    #     # show points in roi
    #     # plt.scatter(xp[:], yp[:], marker='.', color='k', s=1)
    #     # plt.title("Points inside ROI")
    #     # plt.xticks([])
    #     # plt.yticks([])
    #     # plt.show(block=False)
            
        
        


    def timer_callback(self, e):
        self.publisher_.publish(self.gmm_msg)





def main(args=None):
    rospy.init_node("hs_interface")
    hsiObj = ClustersGMM()    
    rospy.spin()
    



if __name__ == '__main__':
    main()
