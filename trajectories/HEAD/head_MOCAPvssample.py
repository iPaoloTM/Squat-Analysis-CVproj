import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes
import sys



##################### ZED

if len(sys.argv) < 2:
    print("usage: sample")
    exit(0)

file_name = sys.argv[1]


if file_name == "sample":
    color = "red"

if file_name == "groundTruth":
    color = "blue"

centered_skeletons = []
barycenter = []
skeletons = ZED_alignment.read_skeletons(file_name + 'ZED')

for joint in skeletons:
    if len(joint)!=0:
        skeleton, center = ZED_alignment.center_skeleton_inBarycenter(np.array(joint))
        centered_skeletons.append(skeleton)
        barycenter.append(center)

print(centered_skeletons)
barycenter = np.mean(np.array(barycenter), axis=0)

############## TEMPORAL ALIGNMENT
pose_index_ = ZED_alignment.main(centered_skeletons)

############## JOINT HEAD  vs
index = 27
list_of_head_coord = []
for i,skeleton in enumerate(centered_skeletons):
    if i > pose_index_[0] and i < pose_index_[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)


##################################################### PLOT
fig_head = plt.figure()
ax = fig_head.add_subplot(111)

list_of_head_coord -= list_of_head_coord[2]
points = np.array(list_of_head_coord)

min_y_index = np.argmin(points[:, 1])
point_with_min_y = points[min_y_index]
print("Point with the minimum y-coordinate:", point_with_min_y)

max_y_index = np.argmax(points[:, 1])
point_with_max_y = points[max_y_index]
print("Point with the maximum y-coordinate:", point_with_max_y)

ax.plot([barycenter[0],barycenter[0]], [point_with_min_y[1], point_with_max_y[1]], c='gray')
ax.scatter(list_of_head_coord[:, 2], list_of_head_coord[:, 1], c=color, label='', s=3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')

plt.savefig('head_ZED_'+file_name+'.png')

# compute ADE
distances = np.abs(list_of_head_coord[:,1] - barycenter[1])
ade = np.mean(distances)
print("ADE:", ade)
