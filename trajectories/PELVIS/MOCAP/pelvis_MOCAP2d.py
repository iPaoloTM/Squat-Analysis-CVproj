import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
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

############### taking 20 points for squat reference
pose_index_gt = []
pose_index_gt, skeletons_gt = MOCAP_alignment.main('groundTruthMOCAP')
print("pose_index_gt: ", pose_index_gt)
# print("skeletons_gt: ", skeletons_gt)
# print("len: ", len(pose_index_gt))
############## JOINT pelvis groundTruth
index = 0
plot_pelvis_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[1] and i < pose_index_gt[19]: # first squat
        print(i)
        plot_pelvis_gt.append(skeleton[index])

plot_pelvis_gt = np.array(plot_pelvis_gt)
# print("plot_pelvis_gt: ",plot_pelvis_gt)
# print("len: ", len(plot_pelvis_gt))

############## JOINT pelvis salienti
interesting_pelvis_coor = []
for i,skeleton in enumerate(pose_index_gt):
    if i > 1 and i < 19:
        interesting_pelvis_coor.append(skeletons_gt[pose_index_gt[i]][0])

interesting_pelvis_coor = np.array(interesting_pelvis_coor)
print("interesting_pelvis_coor: ",interesting_pelvis_coor)
print("len: ", len(interesting_pelvis_coor))




############## taking 20 points for squat sample
pose_index_sample = []
pose_index_sample, skeletons_sample = MOCAP_alignment.main('sampleMOCAP')
print("pose_index_sample: ", pose_index_sample)

############## JOINT pelvis  sample
plot_pelvis_sample = []
for i,skeleton in enumerate(skeletons_sample):
    if i > pose_index_sample[1] and i < pose_index_sample[19]: # first squat
        plot_pelvis_sample.append(skeleton[index])

plot_pelvis_sample = np.array(plot_pelvis_sample)

############## JOINT pelvis salienti
interesting_pelvis_coor_sample = []
for i,skeleton in enumerate(pose_index_sample):
    if i > 1 and i < 19:
        interesting_pelvis_coor_sample.append(skeletons_sample[pose_index_sample[i]][0])

interesting_pelvis_coor_sample = np.array(interesting_pelvis_coor_sample)
#####################################################
coordinates_array = np.array(plot_pelvis_sample)
centroid = np.mean(coordinates_array, axis=0)
centered_coordinates_gt = coordinates_array - centroid

coordinates_array_2 = np.array(plot_pelvis_gt)
centroid_2 = np.mean(coordinates_array_2, axis=0)
centered_coordinates_2 = coordinates_array_2 - centroid_2

##################################################### PLOT
fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111)
# ax.set_title("pelvis FROM MOCAP2d")


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
plt.show()


fig_pelvis_traj = plt.figure()
ax = fig_pelvis_traj.add_subplot(111)
interesting_pelvis_coor -= interesting_pelvis_coor[0]
interesting_pelvis_coor_sample -= interesting_pelvis_coor_sample[0]

ax.scatter(interesting_pelvis_coor[:, 0], interesting_pelvis_coor[:, 1],c='green', label='GROUDTRUTH', s=3)
ax.scatter(interesting_pelvis_coor_sample[:, 0], interesting_pelvis_coor_sample[:, 1],c='orange', label='GROUDTRUTH', s=3)
plt.axis('equal')
plt.show()

############### computing ADE average Distance Error between the points

# Example usage
interesting_pelvis_coor = np.array(interesting_pelvis_coor)
interesting_pelvis_coor_sample = np.array(interesting_pelvis_coor_sample)

ade = compute_ade(interesting_pelvis_coor, interesting_pelvis_coor_sample)
print("ADE:", ade)

fde = compute_fde(interesting_pelvis_coor, interesting_pelvis_coor_sample)
print("FDE:", fde)
plt.savefig('pelvis_MOCAP_groundTruthVSsample.png')
