from h36m_skeleton import H36mSkeleton
from scheletroZED import Poses_3d
import numpy as np
import open3d as o3d
import sys
import os

# Check if there is command-line argument input_file
if len(sys.argv) != 2:
    print("Invalid number of arguments. Please provide exactly one input.")
    sys.exit(1)

input_file = sys.argv[1]

# Create an instance of the Skeleton class -- zed xyz
skeleton = H36mSkeleton()
poses = Poses_3d()
poses_3d = poses.get_poses(input_file=input_file)
body_edges = [
[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[8,10],
[3,11],[11,12],[12,13],[13,14],[14,15],[15,16],[15,17],
[0,18],[18,19],[19,20],[20,32],[32,21],
[0,22],[22,23],[23,24],[24,33],[33,25],
[3,26],[26,27],
[27,28],[28,29],
[27,30],[30,31]]

LFHAND = 19
RHAND = 27
HIP = 0

bone_joint = poses_3d[0]

keypoints = o3d.geometry.PointCloud()
# Set the color for the keypoints
keypoint_color = [1, 0, 0]  # Set the color to red
keypoint_colors = [keypoint_color] * len(bone_joint)
keypoints.points = o3d.utility.Vector3dVector(bone_joint)
keypoints_center = keypoints.get_center()
keypoints.points = o3d.utility.Vector3dVector(bone_joint)
keypoints.colors = o3d.utility.Vector3dVector(keypoint_colors)


skeleton_lines = o3d.geometry.LineSet()
skeleton_color = [0, 0, 0]  # Set the color to red
skeleton_colors = [skeleton_color] * len(body_edges)
skeleton_lines.lines = o3d.utility.Vector2iVector(body_edges)
skeleton_lines.colors = o3d.utility.Vector3dVector(skeleton_colors)

vis = o3d.visualization.Visualizer()

WINDOW_WIDTH=1920
WINDOW_HEIGHT=1080

# Insertion of geometries in the visualizer
vis.create_window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
vis.add_geometry(keypoints)
vis.add_geometry(skeleton_lines)

# Customize visualization settings
opt = vis.get_render_option()
opt.point_size = 10

vis.get_render_option().line_width = 5.0

for i in range(len(poses_3d)):
    # If the measurements are correct the model updates
    if None not in poses_3d[i]:
        new_joints = poses_3d[i]
    else:
        missed_body += 1

    left_hand = new_joints[LFHAND]
    right_hand = new_joints[RHAND]

    skeleton_lines.points = o3d.utility.Vector3dVector(new_joints)
    keypoints.points = o3d.utility.Vector3dVector(new_joints)

    # Update of skeleton
    vis.update_geometry(skeleton_lines)
    vis.update_geometry(keypoints)

    vis.update_renderer()
    vis.poll_events()

vis.run()
