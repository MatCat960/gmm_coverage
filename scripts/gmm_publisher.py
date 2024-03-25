#!/usr/bin/env python3


# -*- coding: utf-8 -*-
"""roadsGMM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fajhstVZ1E4GDuN61exVGDKT9F338-k_
"""

import math
import os
import time, cv2
import tensorflow as tf
from tkinter import filedialog
import numpy as np
from sklearn.mixture import GaussianMixture
# from smooth_predictions_by_belnding_patches import predict_img_with_smooth_windowing
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
from matplotlib import cm

from smoothprediction import predict_img_with_smooth_windowing


# ROS imports
import rospy
import rospkg

from gmm_msgs.msg import GMM, Gaussian
from geometry_msgs.msg import Point

my_dpi = 96
DIM_MT_IMG = [300, 200]

rospack = rospkg.RosPack()

def recompone_images(pat, x, y):
    row = []
    backtoimg = []
    for i in range(len(pat)):
        row.append(np.array(pat[i]))
        if (i+1) % x == 0:
            backtoimg.append(row)
            row = []
    backtoimg = np.array(backtoimg)
    img = unpatchify(backtoimg, (y*256, x*256))
    return img

def predict(name):
    image = cv2.imread(name, cv2.COLOR_BGR2RGB)
    print("---------------------------")
    print("Image size: {}".format(image.shape))

    nr1 = int(image.shape[0] / 256.)
    nc1 = int(image.shape[1] / 256.)
    image = image[0:nr1*256, 0:nc1*256]
    print("Image shape after resizse: {}".format(image.shape))
    if nr1 > nc1:
        n = int( nr1/nc1 )
        image1 = cv2.resize(image, (256,256*n))
    else:
        n = int( nc1/nr1 )
        image1 = cv2.resize(image, (256*n,256))

    nr = int(image1.shape[0] / 256.)
    nc = int(image1.shape[1] / 256.)

    modelpkg = rospack.get_path('gmm_coverage')
    modelpath = os.path.join(modelpkg, 'RoadExtractionModel.h5')
    model = tf.keras.models.load_model(modelpath, compile=False)

    patch_size = 256
    patches = []

    patches_img = patchify(image1, (patch_size, patch_size, 3), step=patch_size)
    for k in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[k,j,:,:]
            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds
            patches.append(single_patch_img)

    #Prediction without using blending patches
    mask_patches = []
    for i in range(len(patches)):
        img = patches[i] / 255.0
        p0 = model.predict(np.expand_dims(img, axis=0))[0][:, :, 0]
        p1 = model.predict(np.expand_dims(np.fliplr(img), axis=0))[0][:, :, 0]
        p1 = np.fliplr(p1)
        p2 = model.predict(np.expand_dims(np.flipud(img), axis=0))[0][:, :, 0]
        p2 = np.flipud(p2)
        p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(img)), axis=0))[0][:, :, 0]
        p3 = np.fliplr(np.flipud(p3))
        thresh = 0.2
        p = (p0 + p1 + p2 + p3) / 4
        mask_patches.append(p)

    prediction = recompone_images(mask_patches, nc, nr)
    pred = (prediction > thresh).astype(np.uint8)

    #Prediction using blending patches
    input_img = image1/255.
    predictions_smooth = predict_img_with_smooth_windowing(
                                                        input_img,
                                                        window_size=patch_size,
                                                        subdivisions=2,
                                                        nb_classes=1,
                                                        pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv)))
                                                        )

    final_prediction = (predictions_smooth > thresh).astype(np.uint8)
    union_prediction = (((prediction + 2*predictions_smooth[:,:,0]) / 2) > thresh).astype(np.uint8)
    return union_prediction*255

def distmt(x, y, img_size, mt):
    dist_x = (x*mt[0])/img_size[1]
    dist_y = (y*mt[1])/img_size[0]
    return [math.sqrt(dist_x**2+dist_y**2), dist_x, dist_y]

# X, Y : meshgrid
def multigauss_pdf(X, Y, means, covariances, weights):
  # Flatten the meshgrid coordinates
    points = np.column_stack([X.flatten(), Y.flatten()])

    # Number of components in the mixture model
    num_components = len(means)

    # Initialize the probabilities
    probabilities = np.zeros_like(X)

    # Calculate the probability for each component
    for i in range(num_components):
        mean = means[i]
        covariance = covariances[i]
        weight = weights[i]

        # Calculate the multivariate Gaussian probability
        exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
        coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
        component_prob = coefficient * np.exp(exponent)

        # Add the component probability weighted by its weight
        probabilities += weight * component_prob.reshape(X.shape)

    return probabilities

xp, yp = [], []

def main():
    rospy.init_node("gmm_predictor")
    pub = rospy.Publisher("/gaussian_mixture_model", GMM, queue_size=10)

    # get params from launch file
    FILENAME = rospy.get_param('~filename', 'desert')
    COMPONENTS_NUM = rospy.get_param('~components_num', 20)
    SHOW_IMAGES = rospy.get_param('~show_images', False)
    SAVE_IMAGES = rospy.get_param('~save_images', False)

    pkgpath = rospack.get_path('gmm_coverage')
    path = os.path.join(pkgpath, 'models', FILENAME)
    file_types = [('Image', '*.jpg;*.png'), ('All files', '*')]
    # name = filedialog.askopenfilename(title='Select an image:', filetypes=file_types, initialdir=path)
    name = os.path.join(path, FILENAME+".png")
    print("Image name : {}".format(name))
    image = predict(name)
    IMG_SIZE = image.shape
    print("Image size: {}".format(IMG_SIZE))
    if SHOW_IMAGES:
        plt.figure(figsize=(IMG_SIZE[0]/my_dpi, IMG_SIZE[1]/my_dpi), dpi=my_dpi)
        # plt.title("Prediction")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray')
        if SAVE_IMAGES:
            plt.savefig("{}_prediction.png".format(FILENAME))
            # files.download("{}_prediction.png".format(FILENAME))


    for i in range(IMG_SIZE[0]):
        for j in range(IMG_SIZE[1]):
            if image[i,j] == 255:
                #xp.append(j)
                #yp.append(IMG_SIZE[1]-i)
                k,x,y = distmt(j, i, IMG_SIZE, DIM_MT_IMG)
                xp.append(x)
                yp.append(DIM_MT_IMG[1]-y)

    #plt.show()

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

    # create ROS Msg
    gmm_msg = GMM()
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

        gmm_msg.gaussians.append(g)
        gmm_msg.weights.append(mix[i])

    
    if (SHOW_IMAGES):
        # generate grid
        # xgrid = np.linspace(0,image.shape[0])
        # ygrid = np.linspace(0,image.shape[1])
        # X, Y = np.meshgrid(xgrid, ygrid)
            #plt.figure(figsize=(IMG_SIZE[0]/my_dpi, IMG_SIZE[1]/my_dpi), dpi=my_dpi)
        plt.figure(figsize=(DIM_MT_IMG[0]/my_dpi, DIM_MT_IMG[1]/my_dpi), dpi=my_dpi)
        plt.scatter(xp[:], yp[:], marker='.', color='k', s=1)
        plt.title("Points inside ROI")
        plt.xticks([])
        plt.yticks([])
        if SAVE_IMAGES:
            plt.savefig("{}_points.png".format(FILENAME))
        #plt.show()

        # plot distribution
            #X = np.arange(0, IMG_SIZE[0], 0.1)
            #Y = np.arange(0, IMG_SIZE[1], 0.1)
        X = np.arange(0, DIM_MT_IMG[0], 0.1)
        Y = np.arange(0, DIM_MT_IMG[1], 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = multigauss_pdf(X, Y, means, covariances, mix)

        # fig, ax = plt.subplots(1, 1, figsize=(12,4), subplot_kw={"projection": "3d"})

        #3D surface plot
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # ax.set_xlabel("X"); ax.set_ylabel("Y")
        # ax.set_zticks([])
        # ax.set_title("Probability density function")
        # ax.invert_xaxis()
        # ax.view_init(60,60,0)

        # 2D heatmap
        fig2, ax2 = plt.subplots(1, 1)

        min = np.min(Z); z_max = np.max(Z)
        c = ax2.pcolormesh(X, Y, Z, cmap='YlOrRd', vmin=z_min, vmax=z_max)
        ax2.set_title("heatmap")
        # fig2.colorbar(c, ax=ax2)
            #ax2.set_xlim([0, IMG_SIZE[0]]); ax2.set_ylabel([0, IMG_SIZE[1]])
        ax2.set_xlim([0, DIM_MT_IMG[0]]); ax2.set_ylim([0, DIM_MT_IMG[1]])
        plt.xticks([]); plt.yticks([])
        if SAVE_IMAGES:
            plt.savefig("{}_heatmap.png".format(FILENAME))
        # files.download("{}_heatmap.png".format(FILENAME))
        plt.show()

    while not rospy.is_shutdown():
        pub.publish(gmm_msg)
        rospy.sleep(10.0)


if __name__ == "__main__":
    main()