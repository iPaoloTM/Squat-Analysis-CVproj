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
# centered_skeletons_gt = []
# barycenter_gt = []
skeletons_gt = MOCAP_alignment.read_skeletons('groundTruthMOCAP')
# skeletons = MOCAP_alignment.read_skeletons('sampleMOCAP')
# for joint in skeletons_gt:
    # if len(joint) > 18 and len(joint)!=0:
        # skeleton, center = MOCAP_3Dplot.center_skeleton(np.array(joint))
        # centered_skeletons_gt.append(skeleton)
        # barycenter_gt.append(center)

# Calculate the average of each coordinate
# barycenter_gt = np.mean(np.array(barycenter_gt), axis=0)
# print(barycenter_gt)
############## TEMPORAL ALIGNMENT
pose_index_gt = MOCAP_alignment.main(skeletons_gt)
############## JOINT pelvis sample vs groundTruth
index = 0
list_of_pelvis_coord_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[2] and i < pose_index_gt[3]:
        list_of_pelvis_coord_gt.append(skeleton[index])

list_of_pelvis_coord_gt = np.array(list_of_pelvis_coord_gt)

##################### sampleMOCAP

############## SPATIAL ALIGNMENT
## get skeletons aligned
centered_skeletons_sample = []
# barycenter_sample = []
skeletons_sample = MOCAP_alignment.read_skeletons('sampleMOCAP')
# for joint in skeletons_sample:
#     if len(joint) > 18 and len(joint)!=0:
#         # skeleton, center = MOCAP_3Dplot.center_skeleton(np.array(joint))
#         centered_skeletons_sample.append(skeleton)
#         barycenter_sample.append(center)

# Calculate the average of each coordinate
# barycenter_sample =
# barycenter_sample = np.mean(np.array(barycenter_sample), axis=0)
# print(barycenter_sample)
############## TEMPORAL ALIGNMENT
pose_index_sample = MOCAP_alignment.main(skeletons_sample)
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

# # Create Open3D point clouds from pelvis data
# point_cloud_sample = o3d.geometry.PointCloud()
# point_cloud_sample.points = o3d.utility.Vector3dVector(centered_coordinates)
# point_cloud_sample.paint_uniform_color([0, 0, 1])  # Set color to blu
#
# point_cloud_gt = o3d.geometry.PointCloud()
# point_cloud_gt.points = o3d.utility.Vector3dVector(centered_coordinates_2)
# point_cloud_gt.paint_uniform_color([1, 0, 0])  # Set color to red
#
# # Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # Add point clouds to the visualization
# vis.add_geometry(point_cloud_sample)
# vis.add_geometry(point_cloud_gt)
#
# # Customize visualization settings
# opt = vis.get_render_option()
# opt.point_size = 5
#
# # Run the visualization
# vis.run()
# vis.destroy_window()

# ##################################################### PLOT
fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111)
# ax.set_title("pelvis FROM MOCAP")

# Add a vertical line
# vertical_line_position = np.mean(list_of_pelvis_coord_sample[:,0])
# ax.axvline(x=vertical_line_position, color='gray', linestyle='-', linewidth=2)
ax.scatter(centered_coordinates[:, 0], centered_coordinates[:, 1], c='red', label='SAMPLE', s=3)
ax.scatter(centered_coordinates_2[:, 0], centered_coordinates_2[:, 1], c='blue', label='GROUNDTRUTH', s=3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')
# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)
# plt.show()

plt.savefig('pelvis_MOCAP_groundTruthVSsample.png')

#
# #####################
# print(list_of_pelvis_coord_sample,list_of_pelvis_coord_gt)
# # Create Open3D point clouds from pelvis data
# point_cloud_sample = o3d.geometry.PointCloud()
# point_cloud_sample.points = o3d.utility.Vector3dVector(list_of_pelvis_coord_gt)
# point_cloud_sample.paint_uniform_color([0, 0, 1])  # Set color to red
#
# point_cloud_gt = o3d.geometry.PointCloud()
# point_cloud_gt.points = o3d.utility.Vector3dVector(list_of_pelvis_coord_sample)
# point_cloud_gt.paint_uniform_color([1, 0, 0])  # Set color to blue
#
# # Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # Add point clouds to the visualization
# vis.add_geometry(point_cloud_sample)
# vis.add_geometry(point_cloud_gt)
#
# # Customize visualization settings
# opt = vis.get_render_option()
# opt.point_size = 5
#
# # Run the visualization
# vis.run()
# vis.destroy_window()
