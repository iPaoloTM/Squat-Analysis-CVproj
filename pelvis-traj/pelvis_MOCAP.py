import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment


pose_index_sample = MOCAP_alignment.main('sampleMOCAP')
pelvises_sample = MOCAP_alignment.read_joint('sampleMOCAP', 'Aliprandi_Girardi:Hip', pose_index_sample[0], pose_index_sample[1])

pose_index_gt = MOCAP_alignment.main('groundTruthMOCAP')
pelvises_gt = MOCAP_alignment.read_joint('groundTruthMOCAP', 'Aliprandi_Girardi:Hip', pose_index_gt[0], pose_index_gt[1])

fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111, projection='3d')
ax.set_title("PELVIS FROM MOCAP")
ax.plot(pelvises_sample[:, 2], pelvises_sample[:, 1], pelvises_sample[:, 0], c='red', label='SAMPLE')
ax.plot(pelvises_gt[:, 2], pelvises_gt[:, 1], pelvises_gt[:, 0], c='blue', label='GT')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_ylabel('Z Label')

# Set legend properties
legend = ax.legend(title='Points', loc='upper right')
plt.show()

#####################

# Create Open3D point clouds from pelvis data
point_cloud_sample = o3d.geometry.PointCloud()
point_cloud_sample.points = o3d.utility.Vector3dVector(pelvises_sample)
point_cloud_sample.paint_uniform_color([1, 0, 0])  # Set color to red

point_cloud_gt = o3d.geometry.PointCloud()
point_cloud_gt.points = o3d.utility.Vector3dVector(pelvises_gt)
point_cloud_gt.paint_uniform_color([0, 0, 1])  # Set color to blue

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add point clouds to the visualization
vis.add_geometry(point_cloud_sample)
vis.add_geometry(point_cloud_gt)

# Customize visualization settings
opt = vis.get_render_option()
opt.point_size = 10

# Run the visualization
vis.run()
vis.destroy_window()
