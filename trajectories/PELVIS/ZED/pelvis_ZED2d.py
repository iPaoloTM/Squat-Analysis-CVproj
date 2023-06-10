import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes


def compute_ade(list1, list2):
    assert len(list1) == len(list2), "Point lists must have the same length."

    distances = np.linalg.norm(list1 - list2, axis=1)  # Compute Euclidean distances
    ade = np.mean(distances)  # Compute average distance

    return ade


############### taking 20 points for squat reference

pose_index_gt, skeletons_gt = ZED_alignment.main('groundTruthZED')
# print("pose_index_gt: ", pose_index_gt)
print("skeletons_gt: ", skeletons_gt)

print("len: ", len(pose_index_gt))
############## JOINT pelvis groundTruth
index = 0
plot_pelvis_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[0] and i < pose_index_gt[21]: # first squat
        plot_pelvis_gt.append(skeleton[index])

plot_pelvis_gt = np.array(plot_pelvis_gt)
# print("plot_pelvis_gt: ",plot_pelvis_gt)
# print("len: ", len(plot_pelvis_gt))

############## JOINT pelvis salienti
interesting_pelvis_coor = []
for i,skeleton in enumerate(pose_index_gt):
    if i > 0 and i < 21:
        interesting_pelvis_coor.append(skeletons_gt[pose_index_gt[i]][0])

interesting_pelvis_coor = np.array(interesting_pelvis_coor)
print("interesting_pelvis_coor: ",interesting_pelvis_coor)
print("len: ", len(interesting_pelvis_coor))




############## taking 20 points for squat sample
pose_index_sample, skeletons_sample = ZED_alignment.main('sampleZED')
############## JOINT pelvis  sample
plot_pelvis_sample = []
for i,skeleton in enumerate(skeletons_sample):
    if i > pose_index_sample[0] and i < pose_index_sample[21]: # first squat
        plot_pelvis_sample.append(skeleton[index])

plot_pelvis_sample = np.array(plot_pelvis_sample)

############## JOINT pelvis salienti
interesting_pelvis_coor_sample = []
for i,skeleton in enumerate(pose_index_sample):
    if i > 0 and i < 21:
        interesting_pelvis_coor_sample.append(skeletons_sample[pose_index_sample[i]][0])

interesting_pelvis_coor_sample = np.array(interesting_pelvis_coor_sample)
#####################################################
coordinates_array = np.array(plot_pelvis_sample)
centroid = np.mean(coordinates_array, axis=0)
centered_coordinates = coordinates_array - centroid

coordinates_array_2 = np.array(plot_pelvis_gt)
centroid_2 = np.mean(coordinates_array_2, axis=0)
centered_coordinates_2 = coordinates_array_2 - centroid_2

##################################################### PLOT
fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111)
# ax.set_title("pelvis FROM zed2d")


centered_coordinates -= centered_coordinates[0]
# y_values = centered_coordinates[:, 1] # y values for coloring
# y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
# cmap = cm.get_cmap('Reds')
# ax.scatter(centered_coordinates[:, 0], centered_coordinates[:, 1], c=cmap(y_values_norm), label='SAMPLE', s=3)
ax.scatter(centered_coordinates[:, 0], centered_coordinates[:, 1], c='red', label='SAMPLE', s=3)



centered_coordinates_2 -= centered_coordinates_2[0]
# y_values = centered_coordinates_2[:, 1] # y values for coloring
# y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
# cmap = cm.get_cmap('Blues')
# ax.scatter(centered_coordinates_2[:, 0], centered_coordinates_2[:, 1],c=cmap(y_values_norm), label='GROUDTRUTH', s=3)
ax.scatter(centered_coordinates_2[:, 0], centered_coordinates_2[:, 1],c='blue', label='GROUDTRUTH', s=3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')
# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)
# plt.show()


ax.scatter(interesting_pelvis_coor[:, 0], interesting_pelvis_coor[:, 1],c='green', label='GROUDTRUTH', s=3)
ax.scatter(interesting_pelvis_coor_sample[:, 0], interesting_pelvis_coor_sample[:, 1],c='orange', label='GROUDTRUTH', s=3)

plt.savefig('pelvis_ZED_groundTruthVSsample.png')




############### computing ADE average Distance Error between the points

# Example usage
interesting_pelvis_coor = np.array(interesting_pelvis_coor)
interesting_pelvis_coor_sample = np.array(interesting_pelvis_coor_sample)

ade = compute_ade(interesting_pelvis_coor, interesting_pelvis_coor_sample)
print("ADE:", ade)
