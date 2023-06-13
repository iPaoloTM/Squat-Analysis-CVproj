import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes
import matplotlib.colors as mcolors


def compute_ade(list1, list2):
    assert len(list1) == len(list2), "Point lists must have the same length."

    distances = np.linalg.norm(list1 - list2, axis=1)  # Compute Euclidean distances
    ade = np.mean(distances)  # Compute average distance

    return ade


def compute_fde(predicted_trajectory, ground_truth_trajectory):
    # Extract the final positions
    predicted_final_pos = predicted_trajectory[-1]
    ground_truth_final_pos = ground_truth_trajectory[-1]

    fde_error = np.linalg.norm(predicted_final_pos - ground_truth_final_pos)

    return fde_error


def center_and_translate_coordinates(coordinates):
    # Find the mean of x-coordinates
    mean_x = np.mean(coordinates[:, 0])

    # Center x-coordinates around 0
    coordinates[:, 0] -= mean_x

    # Determine the z-coordinate of the third element
    z_reference = coordinates[2, 2]

    # Translate z-coordinates
    coordinates[:, 2] -= z_reference

    return coordinates


############### taking 20 points for squat reference
pose_index_gt = []
pose_index_gt, skeletons_gt = ZED_alignment.main('groundTruthZED')
print("pose_index_gt: ", pose_index_gt)
# print("skeletons_gt: ", skeletons_gt)
# print("len: ", len(pose_index_gt))
############## JOINT head groundTruth
index = 4
plot_head_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[1] and i < pose_index_gt[19]: # first squat
        print(i)
        plot_head_gt.append(skeleton[index])

plot_head_gt = np.array(plot_head_gt)
# print("plot_head_gt: ",plot_head_gt)
# print("len: ", len(plot_head_gt))

############## JOINT head salienti
interesting_head_coor = []
for i,skeleton in enumerate(pose_index_gt):
    if i > 1 and i < 19:
        interesting_head_coor.append(skeletons_gt[pose_index_gt[i]][0])

interesting_head_coor = np.array(interesting_head_coor)
print("interesting_head_coor: ",interesting_head_coor)
print("len: ", len(interesting_head_coor))




############## taking 20 points for squat sample
pose_index_sample = []
pose_index_sample, skeletons_sample = ZED_alignment.main('sampleZED')
print("pose_index_sample: ", pose_index_sample)

############## JOINT head  sample
plot_head_sample = []
for i,skeleton in enumerate(skeletons_sample):
    if i > pose_index_sample[1] and i < pose_index_sample[19]: # first squat
        plot_head_sample.append(skeleton[index])

plot_head_sample = np.array(plot_head_sample)

############## JOINT head salienti
interesting_head_coor_sample = []
for i,skeleton in enumerate(pose_index_sample):
    if i > 1 and i < 19:
        interesting_head_coor_sample.append(skeletons_sample[pose_index_sample[i]][0])

interesting_head_coor_sample = np.array(interesting_head_coor_sample)
#####################################################
coordinates_array = np.array(plot_head_sample)
centroid = np.mean(coordinates_array, axis=0)
centered_coordinates_gt = coordinates_array - centroid

coordinates_array_2 = np.array(plot_head_gt)
centroid_2 = np.mean(coordinates_array_2, axis=0)
centered_coordinates_2 = coordinates_array_2 - centroid_2

##################################################### PLOT
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
# ax.set_title("head FROM ZED2d")


centered_coordinates_gt -= centered_coordinates_gt[0]
# Define the colormap without white color
colors = [ 'darkred', 'red']
# Calculate the minimum and maximum values
y_min_gt = np.min(centered_coordinates_gt)
y_max_gt = np.max(centered_coordinates_gt)
# Define the colors for the two sections
color_start = 'red'
color_end = 'darkred'

# Create the colormap with two sections
bounds = [y_min_gt, y_max_gt]
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
y_values = centered_coordinates_gt[:, 1] # y values for coloring
y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
norm = mcolors.Normalize(vmin=y_min_gt, vmax=y_max_gt)




points = np.array(centered_coordinates_gt)

# Find the index of the point with the minimum y-coordinate
min_y_index = np.argmin(points[:, 1])
point_with_min_y = points[min_y_index]
print("Point with the minimum y-coordinate:", point_with_min_y)

# Find the index of the point with the maximum y-coordinate
max_y_index = np.argmax(points[:, 1])
point_with_max_y = points[max_y_index]
print("Point with the maximum y-coordinate:", point_with_max_y)

ax.plot([0,0], [point_with_min_y[1], point_with_max_y[1]], c='gray')


ax.scatter(centered_coordinates_gt[:, 0], centered_coordinates_gt[:, 1],  c=centered_coordinates_gt[:, 1], cmap=cmap, norm=norm, label='Data', s=3)


centered_coordinates_2 -= centered_coordinates_2[0]
colors_2 = [ 'darkblue', 'cyan']
y_min = np.min(centered_coordinates_2)
y_max = np.max(centered_coordinates_2)
bounds = [y_min, y_max]
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors_2)
y_values = centered_coordinates_2[:, 1] # y values for coloring
y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
norm = mcolors.Normalize(vmin=y_min, vmax=y_max)
plt.scatter(centered_coordinates_2[:, 0], centered_coordinates_2[:, 1], c=centered_coordinates_2[:, 1], cmap=cmap, norm=norm,  label='GROUDTRUTH', s=3)
cbar = plt.colorbar()



ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')
# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)
# plt.show()
plt.savefig('head_ZED_groundTruthVSsample.png')

fig_head_traj = plt.figure()
ax = fig_head_traj.add_subplot(111)
interesting_head_coor -= interesting_head_coor[0]
interesting_head_coor_sample -= interesting_head_coor_sample[0]

ax.scatter(interesting_head_coor[:, 0], interesting_head_coor[:, 1],c='blue', label='GROUDTRUTH', s=3)
ax.scatter(interesting_head_coor_sample[:, 0], interesting_head_coor_sample[:, 1],c='red', label='GROUDTRUTH', s=3)
plt.axis('equal')
# plt.show()
plt.savefig('head_ZED_groundTruthVSsample_sampledTraj.png')

############### computing ADE average Distance Error between the points

# Example usage
interesting_head_coor = np.array(interesting_head_coor)
interesting_head_coor_sample = np.array(interesting_head_coor_sample)

ade = compute_ade(interesting_head_coor, interesting_head_coor_sample)
print("ADE:", ade)

fde = compute_fde(interesting_head_coor, interesting_head_coor_sample)
print("FDE:", fde)
