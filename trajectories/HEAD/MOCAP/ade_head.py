import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes
import ZED_3Dplot

##################### groundTruthZED
skeletons_gt = ZED_alignment.read_skeletons('groundTruthZED')
############## TEMPORAL ALIGNMENT
pose_index_gt = ZED_alignment.main(skeletons_gt)
############## JOINT pelvis sample vs groundTruth
index = 0
list_of_pelvis_coord_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[4] and i < pose_index_gt[5]:
        list_of_pelvis_coord_gt.append(skeleton[index])

list_of_pelvis_coord_gt = np.array(list_of_pelvis_coord_gt)

##################### sampleZED




############## SPATIAL ALIGNMENT
skeletons_sample = ZED_alignment.read_skeletons('sampleZED')

############## TEMPORAL ALIGNMENT
pose_index_sample = ZED_alignment.main(skeletons_sample)
############## JOINT pelvis sample vs groundTruth
index = 0
list_of_pelvis_coord_sample = []
for i,skeleton in enumerate(skeletons_sample):
    if i > pose_index_sample[4] and i < pose_index_sample[5]:
        list_of_pelvis_coord_sample.append(skeleton[index])

list_of_pelvis_coord_sample = np.array(list_of_pelvis_coord_sample)

#####################
print(list_of_pelvis_coord_sample,list_of_pelvis_coord_gt)

coordinates_array = np.array(list_of_pelvis_coord_sample)
centroid = np.mean(coordinates_array, axis=0)
centered_coordinates = coordinates_array - centroid

coordinates_array_2 = np.array(list_of_pelvis_coord_gt)
centroid_2 = np.mean(coordinates_array_2, axis=0)
centered_coordinates_2 = coordinates_array_2 - centroid_2

#####################
##################################################### PLOT
fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111)
# ax.set_title("pelvis FROM zed2d")

y_values = centered_coordinates[:, 1] # y values for coloring
y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
cmap = cm.get_cmap('Reds')
ax.scatter(centered_coordinates[:, 0], centered_coordinates[:, 1], c=cmap(y_values_norm), label='SAMPLE', s=3)

y_values = centered_coordinates_2[:, 1] # y values for coloring
y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
cmap = cm.get_cmap('Blues')
ax.scatter(centered_coordinates_2[:, 0], centered_coordinates_2[:, 1],c=cmap(y_values_norm), label='GROUDTRUTH', s=3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')
plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)
# plt.show()
plt.savefig('pelvis_ZED_groundTruthVSsample.png')
