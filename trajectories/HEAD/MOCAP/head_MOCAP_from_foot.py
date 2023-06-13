import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
from scipy.spatial import procrustes
import sys


##################### MOCAP

if len(sys.argv) < 2:
    print("usage: python head_MOCAP_from_foot.py <FILE>")
    exit(0)

file_name = sys.argv[1]


if file_name == "sample":
    color = "red"

if file_name == "groundTruth":
    color = "blue"

centered_skeletons = []
barycenter = []
skeletons = MOCAP_alignment.read_skeletons(file_name + 'MOCAP')
for joint in skeletons:
    if len(joint) > 18 and len(joint)!=0:
        skeleton, center = MOCAP_alignment.center_skeleton_inBarycenter(np.array(joint))
        centered_skeletons.append(skeleton)
        barycenter.append(center)

# Calculate the average of each coordinate
barycenter = np.mean(np.array(barycenter), axis=0)
############## TEMPORAL ALIGNMENT
pose_index_ = MOCAP_alignment.main(centered_skeletons)
############## JOINT HEAD  vs
index = 4
list_of_head_coord = []
for i,skeleton in enumerate(centered_skeletons):
    if i > pose_index_[0] and i < pose_index_[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)

##################################################### PLOT
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
# ax.set_title("head FROM MOCAP")

list_of_head_coord -= list_of_head_coord[2]
list_of_head_coord += barycenter[2]

points = np.array(list_of_head_coord)

# Find the index of the point with the minimum y-coordinate
min_y_index = np.argmin(points[:, 1])
point_with_min_y = points[min_y_index]
print("Point with the minimum y-coordinate:", point_with_min_y)

# Find the index of the point with the maximum y-coordinate
max_y_index = np.argmax(points[:, 1])
point_with_max_y = points[max_y_index]
print("Point with the maximum y-coordinate:", point_with_max_y)

ax.plot([barycenter[2], barycenter[2]], [point_with_min_y[1], point_with_max_y[1]], c='gray')
# ax.scatter(list_of_head_coord[:, 1], list_of_head_coord[:, 2], c=color, label='', s=3)
ax.scatter(list_of_head_coord[:, 2], list_of_head_coord[:, 1], c=color, label='', s=3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')
# plt.xlim(barycenter[0]-1,barycenter[0]+1)

plt.savefig('head_MOCAP_'+file_name+'.png')

# compute ADE
distances = np.abs(list_of_head_coord[:,1] - barycenter[1])
ade = np.mean(distances)
print("ADE:", ade)
# ####################################################### PLOT from upper
# fig_head = plt.figure()
# ax = fig_head.add_subplot(111)
#
# num_circles = 3
# outer_radius = 15
# center_x = 0
# center_y = 0
#
# # Define the colors for each area
# colors = [[1.0, 0.6, 0.6], [1.0, 0.8, 0.6], [0.7, 0.9, 0.7]]
#
# # Plot the concentric circles
# for i in range(num_circles):
#     radius = outer_radius * (num_circles - i) / num_circles
#     color_circle = colors[i % len(colors)]  # Cycle through the colors
#     circle = plt.Circle((barycenter[0], barycenter[0]), radius, color=color_circle, fill=True)
#     ax.add_patch(circle)
#
# ax.set_xlim(center_x - outer_radius, center_x + outer_radius)
# ax.set_ylim(center_y - outer_radius, center_y + outer_radius)
#
# ax.scatter(list_of_head_coord[:, 0], list_of_head_coord[:, 2], c=color, label='')
#
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# plt.axis('equal')
#
# plt.savefig('head_up_MOCAP_'+file_name+'.png')
