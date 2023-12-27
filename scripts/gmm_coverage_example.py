import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path

from sklearn.mixture import GaussianMixture

ROBOTS_NUM = 12
ROBOT_RANGE = 10.0
TARGETS_NUM = 4
COMPONENTS_NUM = 4
PARTICLES_NUM = 500
AREA_W = 40.0
vmax = 1.5


def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points

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

"""## Define environment and targets"""

targets = np.zeros((TARGETS_NUM, 1, 2))
for i in range(TARGETS_NUM):
  targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
  targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)

plt.plot([-0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, -0.5*AREA_W], c='tab:blue', label="Environment")
plt.plot([0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
plt.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
plt.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='tab:blue')
plt.scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', label="Targets")
# plt.legend()
plt.show()

"""## Define GMM from noisy target measurements"""

STD_DEV = 0.5
samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 2))
for k in range(TARGETS_NUM):
  for i in range(PARTICLES_NUM):
    samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 2)

plt.plot([-0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, -0.5*AREA_W], c='tab:blue', label="Environment")
plt.plot([0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
plt.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
plt.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='tab:blue')
plt.scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', label="Targets")
plt.scatter(samples[:, :, 0], samples[:, :, 1], c='tab:olive')

# Fit GMM
samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 2))
print(samples.shape)
gmm = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
gmm.fit(samples)

means = gmm.means_
covariances = gmm.covariances_
mix = gmm.weights_

print(f"Means: {means}")
print(f"Covs: {covariances}")
print(f"Mix: {mix}")

covariances[0]

# ROBOTS_NUM = np.random.randint(6, ROBOTS_MAX)
ROBOTS_NUM = 8
converged = False
NUM_STEPS = 100
GAUSS_PT = np.zeros((1, 2))
GAUSS_COV = 2.0*np.eye(2)
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
robots_hist[0, :, :] = points
vis_regions = []
discretize_precision = 0.5
# fig, axs = plt.subplots(2, int(NUM_STEPS/2), figsize=(18,5))
for s in range(1, NUM_STEPS+1):
  row = 0
  if s > 5:
    row = 1

  # mirror points across each edge of the env
  dummy_points = np.zeros((5*ROBOTS_NUM, 2))
  dummy_points[:ROBOTS_NUM, :] = points
  mirrored_points = mirror(points)
  mir_pts = np.array(mirrored_points)
  dummy_points[ROBOTS_NUM:, :] = mir_pts

  # Voronoi partitioning
  vor = Voronoi(dummy_points)

  conv = True
  lim_regions = []
  for idx in range(ROBOTS_NUM):
    region = vor.point_region[idx]
    poly_vert = []
    for vert in vor.regions[region]:
      v = vor.vertices[vert]
      poly_vert.append(v)
      # plt.scatter(v[0], v[1], c='tab:red')

    poly = Polygon(poly_vert)
    x,y = poly.exterior.xy
    # plt.plot(x, y, c='tab:orange')
    # robot = np.array([-18.0, -12.0])
    robot = vor.points[idx]

    # plt.scatter(robot[0], robot[1])

    # Intersect with robot range
    step = 0.5
    range_pts = []
    for th in np.arange(0.0, 2*np.pi, step):
      xi = robot[0] + ROBOT_RANGE * np.cos(th)
      yi = robot[1] + ROBOT_RANGE * np.sin(th)
      pt = Point(xi, yi)
      range_pts.append(pt)
      # plt.plot(xi, yi, c='tab:blue')

    range_poly = Polygon(range_pts)
    xc, yc = range_poly.exterior.xy

    lim_region = intersection(poly, range_poly)
    lim_regions.append(lim_region)

    # Calculate centroid with gaussian distribution
    xmin, ymin, xmax, ymax = lim_region.bounds
    # print(f"x range: {xmin} - {xmax}")
    # print(f"y range: {ymin} - {ymax}")
    A = 0.0
    Cx = 0.0; Cy = 0.0
    dA = discretize_precision ** 2
    # pts = [Point(xmin, ymin), Point(xmax, ymin), Point(xmax, ymax), Point(xmin, ymax)]
    # bound = Polygon(pts)
    for i in np.arange(xmin, xmax, discretize_precision):
      for j in np.arange(ymin, ymax, discretize_precision):
        pt_i = Point(i,j)
        if lim_region.contains(pt_i):
          dA_pdf = dA * gmm_pdf(i, j, means, covariances, mix)
          # print(dA_pdf)
          A = A + dA_pdf
          Cx += i*dA_pdf
          Cy += j*dA_pdf

    Cx = Cx / A
    Cy = Cy / A



    # centr = np.array([lim_region.centroid.x, lim_region.centroid.y])
    centr = np.array([Cx, Cy]).transpose()
    # print(f"Robot: {robot}")
    # print(f"Centroid: {centr}")
    dist = np.linalg.norm(robot-centr)
    vel = 0.8 * (centr - robot)
    vel[0, 0] = max(-vmax, min(vmax, vel[0,0]))
    vel[0, 1] = max(-vmax, min(vmax, vel[0,1]))

    points[idx, :] = robot + vel
    if dist > 0.1:
      conv = False

  # Save positions for visualization
  if s == 1:
    vis_regions.append(lim_regions)
  robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
  vis_regions.append(lim_regions)

  if conv:
    print(f"Converged in {s} iterations")
    break
  # axs[row, s-1-5*row].scatter(points[:, 0], points[:, 1])



# plt.scatter(points[:, 0], points[:, 1])
# for region in lim_regions:
#   x,y = region.exterior.xy
#   plt.plot(x, y, c="tab:red")

Cx

# points = -0.5*AREA_W + 0.5*AREA_W * np.random.rand(ROBOTS_NUM, 2)
# points[:, 1] = -points[:, 1]
# points[0]

COLS = 3
fig, axs = plt.subplots(2, COLS, figsize=(18,5))
vis_step = int((robots_hist.shape[0])/COLS/2)
print(vis_step)
print(0.5*robots_hist.shape[0] % COLS)
if (0.5*robots_hist.shape[0] % COLS > 0.5):
  vis_step += 1
print(f"Visualization step: {vis_step}")
print(f"Robots hist shape : {robots_hist.shape[0]}")
count = 0
for i in range(0, robots_hist.shape[0], vis_step):
  r = robots_hist[i, :, :]
  region_i = vis_regions[i]
  row = 0
  if count > COLS-1:
    row = 1

  area_tot = 0.0
  samples = samples.reshape(TARGETS_NUM, PARTICLES_NUM, 2)
  for j in range(ROBOTS_NUM):
    axs[row, count-5*row].scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', marker='*')
    axs[row, count-5*row].scatter(r[j, 0], r[j, 1], c='tab:blue')
    xi, yi = region_i[j].exterior.xy
    area_tot += region_i[j].area
    axs[row, count-5*row].plot(xi, yi, c="tab:red")
  # axs[row, count-5*row].set_xticks([]); axs[row, count-5*row].set_yticks([])
  axs[row, count-5*row].set_title(f"t = {i}")
  count += 1

plt.show()



# Plot final configuration
# axs[-1, -1].scatter(points[:, 0], points[:, 1], c='tab:blue')
# axs[-1, -1].set_xticks([]); axs[-1, -1].set_yticks([])
# axs[-1, -1].set_title("Final config.")
# for region in lim_regions:
#   x,y = region.exterior.xy
#   axs[-1, -1].plot(x, y, c="tab:red")

area_max = ROBOTS_NUM * np.pi * ROBOT_RANGE ** 2
r = area_tot / area_max

print("Area covered: {}".format(area_tot))
print("ratio: {}".format(r))