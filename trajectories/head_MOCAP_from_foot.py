import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
from scipy.spatial import procrustes
import MOCAP_3Dplot

##################### groundTruthMOCAP

############## SPATIAL ALIGNMENT
## get skeletons aligned
centered_skeletons = []
barycenter = []
skeletons = MOCAP_alignment.read_skeletons('groundTruthMOCAP')
# skeletons = MOCAP_alignment.read_skeletons('groundTruthMOCAP')
for joint in skeletons:
    if len(joint) > 18 and len(joint)!=0:
        skeleton, center = MOCAP_3Dplot.center_skeleton(np.array(joint))
        centered_skeletons.append(skeleton)
        barycenter.append(center)

# Calculate the average of each coordinate
barycenter = np.mean(np.array(barycenter), axis=0)
# print(barycenter)
############## TEMPORAL ALIGNMENT
pose_index_groundTruth = MOCAP_alignment.main(centered_skeletons)
############## JOINT HEAD groundTruth vs groundTruth
index = 3
list_of_head_coord = []
for i,skeleton in enumerate(centered_skeletons):
    if i > pose_index_groundTruth[0] and i < pose_index_groundTruth[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)

##################################################### PLOT
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
# ax.set_title("head FROM MOCAP")

# Add a vertical line
vertical_line_position = np.mean(list_of_head_coord[:,0])
ax.axvline(x=vertical_line_position, color='gray', linestyle='-', linewidth=2)
ax.scatter(list_of_head_coord[:, 0], list_of_head_coord[:, 1], c='red', s=3)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
plt.axis('equal')
plt.xlim(vertical_line_position-1, vertical_line_position+1)
plt.ylim(0, 1.75)
# plt.show()

plt.savefig('head_MOCAP_groundTruth.png')
########### PLOT from upper
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
# ax.set_title("head from upper FROM MOCAP")

# Define the number of concentric circles
num_circles = 3
# 5 cm diametro

# Set the radius of the outermost circle
outer_radius = 15

# Set the center coordinates
center_x = 0
center_y = 0

# Define the colors for each area
colors = [[1.0, 0.6, 0.6], [1.0, 0.8, 0.6], [0.7, 0.9, 0.7]]

# Plot the concentric circles
for i in range(num_circles):
    radius = outer_radius * (num_circles - i) / num_circles
    color = colors[i % len(colors)]  # Cycle through the colors
    circle = plt.Circle((barycenter[0], barycenter[2]), radius, color=color, fill=True)
    ax.add_patch(circle)

# Set the aspect ratio to be equal
ax.set_aspect('equal')

# Set the limits of the plot
ax.set_xlim(center_x - outer_radius, center_x + outer_radius)
ax.set_ylim(center_y - outer_radius, center_y + outer_radius)
ax.scatter(list_of_head_coord[:, 0], list_of_head_coord[:, 2], c='red')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
plt.axis('equal')
ax.set_aspect('equal')

# Set legend properties
# legend = ax.legend(title='Points', loc='upper right')

# Barycenter coordinates
barycenter = np.array([barycenter[0], barycenter[2]])
# ax.scatter(barycenter[0], barycenter[1], c='blue', label='barycenter')

distances = []  # 0 2
# Compute Euclidean distance for each point
for point in list_of_head_coord:
    point_coords = np.array([point[0],point[2]])
    distance = np.linalg.norm(point_coords - barycenter) # x y mocap x z zed
    distances.append(distance)

# print(distances)
print("DISTANCE FROM BARYCENTER : ", np.mean(distances), " mm")
# plt.show()
plt.savefig('head_up_MOCAP_groundTruth.png')
