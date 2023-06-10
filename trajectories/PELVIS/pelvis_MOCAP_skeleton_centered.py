import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
from scipy.spatial import procrustes
import MOCAP_3Dplot

##################### sampleMOCAP

############## SPATIAL ALIGNMENT
## get skeletons aligned
centered_skeletons = []
skeletons = MOCAP_alignment.read_skeletons('sampleMOCAP')
for joint in skeletons:
    if len(joint) > 18 and len(joint)!=0:
        skeleton = MOCAP_3Dplot.center_skeleton(np.array(joint))
        centered_skeletons.append(skeleton)

############## TEMPORAL ALIGNMENT
pose_index_sample = MOCAP_alignment.main(centered_skeletons)

print("-----")

############## JOINT PELVIS sample vs groundTruth
index = 0
list_of_head_coord = []
for i,skeleton in enumerate(centered_skeletons):
    if i > pose_index_sample[0] and i < pose_index_sample[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)


#####################

# Create Open3D point clouds from pelvis data
point_cloud_sample = o3d.geometry.PointCloud()
point_cloud_sample.points = o3d.utility.Vector3dVector(list_of_head_coord)
point_cloud_sample.paint_uniform_color([1, 0, 0])  # Set color to red

# point_cloud_gt = o3d.geometry.PointCloud()
# point_cloud_gt.points = o3d.utility.Vector3dVector(pelvises_gt)
# point_cloud_gt.paint_uniform_color([0, 0, 1])  # Set color to blue

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add point clouds to the visualization
vis.add_geometry(point_cloud_sample)
# vis.add_geometry(point_cloud_gt)

# Customize visualization settings
opt = vis.get_render_option()
opt.point_size = 10

# Run the visualization
vis.run()
vis.destroy_window()
