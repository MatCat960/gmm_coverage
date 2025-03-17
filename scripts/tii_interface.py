#!/usr/bin/env python3

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import yaml
from roipoly import RoiPoly

# ROS imports
from ament_index_python.packages import get_package_prefix, get_package_share_directory

config_dir = os.path.join(get_package_share_directory('gmm_coverage'), 'config')
config_file = os.path.join(config_dir, 'polygon_gmm.yaml')
with open(config_file, "r") as f:
    data = yaml.safe_load(f)

AREA_X = data["/**"]["ros__parameters"]["area_size_x"]
AREA_Y = data["/**"]["ros__parameters"]["area_size_y"]
AREA_LEFT = data["/**"]["ros__parameters"]["area_left"]
AREA_BOTTOM = data["/**"]["ros__parameters"]["area_bottom"]
COMPONENTS_NUM = data["/**"]["ros__parameters"]["components_num"]


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


def main():
    # find blank image
    path = os.path.join(get_package_share_directory('gmm_coverage'), 'scripts')
    img_path = os.path.join(path, "blank.jpg")

    # show graphical interface
    image = plt.imread(img_path)
    plt.xlim([0.0, AREA_X])
    plt.ylim([0.0, AREA_Y])
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)

    # get ROI
    roi = RoiPoly(color='r')
    coords = roi.get_roi_coordinates()
    xr = []             # x coordinates of ROI
    yr = []             # y coordinates of ROI
    for tp in coords:
        xr.append(tp[0])
        yr.append(tp[1])

    # get min and max value of x and y
    xmin = min(xr)
    xmax = max(xr)
    ymin = min(yr)
    ymax = max(yr)

    # generate 4000 points in ROI
    xp = []
    yp = []
    cnt = 0
    while cnt < 4000:
        xt = xmin + np.random.random()*(xmax-xmin)
        yt = ymin + np.random.random()*(ymax-ymin)
        if isInside(xt, yt, xr, yr):
            xp.append(xt)
            yp.append(yt)
            cnt += 1

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
        m[0] += AREA_LEFT
        m[1] += AREA_BOTTOM

    print("cov shape: ", covariances.shape)
    print("Means: {}".format(means))
    print("Coveriances: {}".format(covariances))
    print("Mixture proportions: {}".format(mix))

    xg = np.linspace(AREA_LEFT, AREA_X+AREA_LEFT, 100)
    yg = np.linspace(AREA_BOTTOM, AREA_Y+AREA_BOTTOM, 100)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = gmm_pdf(Xg, Yg, means, covariances, mix)
    Z = Z.reshape(100, 100)

    config_file = 'polygon_gmm.yaml'
    # write to file
    covs = covariances.reshape(-1, 4)
    data["/**"]["ros__parameters"]["gaussians_x"] = means[:, 0].tolist()
    data["/**"]["ros__parameters"]["gaussians_y"] = means[:, 1].tolist()
    data["/**"]["ros__parameters"]["gaussians_xx"] = covs[:, 0].tolist()
    data["/**"]["ros__parameters"]["gaussians_xy"] = covs[:, 1].tolist()
    data["/**"]["ros__parameters"]["gaussians_yx"] = covs[:, 2].tolist()
    data["/**"]["ros__parameters"]["gaussians_yy"] = covs[:, 3].tolist()
    data["/**"]["ros__parameters"]["mix"] = mix.tolist()
    with open(config_file, "w") as f:
      yaml.dump(data, f, default_flow_style=False)

    # plot
    fig, ax = plt.subplots()
    ax.pcolormesh(Xg, Yg, Z, cmap="YlOrRd")
    plt.show()

if __name__ == '__main__':
    main()
