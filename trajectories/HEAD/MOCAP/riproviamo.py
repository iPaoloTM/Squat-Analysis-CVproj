import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
from scipy.spatial import procrustes
import sys


##################### MOCAP

####################################################### sample

centered_skeletons = []
barycenter = []
skeletons = MOCAP_alignment.read_skeletons('sampleMOCAP')
pose_index_ = MOCAP_alignment.main(skeletons)
############## JOINT HEAD  vs
index = 4
list_of_head_coord = []
for i,skeleton in enumerate(skeletons):
    if i > pose_index_[0] and i < pose_index_[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)


list_of_head_coord -= list_of_head_coord[2]

####################################################### gt

centered_skeletons_gt = []
barycenter_gt = []
skeletons_gt = MOCAP_alignment.read_skeletons('groundTruthMOCAP')
pose_index__gt = MOCAP_alignment.main(skeletons_gt)
############## JOINT HEAD  vs
list_of_head_coord_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_[0] and i < pose_index_[1]:
        list_of_head_coord_gt.append(skeleton[index])

list_of_head_coord_gt = np.array(list_of_head_coord_gt)


list_of_head_coord_gt -= list_of_head_coord_gt[2]
####################################################### PLOT from upper




fig_head = plt.figure()
ax = fig_head.add_subplot(111)

num_circles = 3
outer_radius = 1
center_x = 0
center_y = 0

# Define the colors for each area
colors = [[1.0, 0.6, 0.6], [1.0, 0.8, 0.6], [0.7, 0.9, 0.7]]

# # Plot the concentric circles
# for i in range(num_circles):
#     radius = outer_radius * (num_circles - i) / num_circles
#     color_circle = colors[i % len(colors)]  # Cycle through the colors
#     circle = plt.Circle((0,0), radius, color=color_circle, fill=True)
#     ax.add_patch(circle)
#
# ax.set_xlim(center_x - outer_radius, center_x + outer_radius)
# ax.set_ylim(center_y - outer_radius, center_y + outer_radius)

ax.scatter(list_of_head_coord[:, 2], list_of_head_coord[:, 0], c='red', label='', s=3)
ax.scatter(list_of_head_coord_gt[:, 2], list_of_head_coord_gt[:, 0], c='blue', label='', s=3)

plt.axis('equal')
plt.show()
# plt.savefig('head_up_MOCAP_samplevsGroundtruth.png')
