"""Example code for importing a single rigid body trajectory into Rhino from a Optitrack CSV file.

Copyright (c) 2016, Garth Zeglin.  All rights reserved. Licensed under the terms
of the BSD 3-clause license as included in LICENSE.

Example code for generating a path of Rhino 'planes' (e.g. coordinate frame)
from a trajectory data file.  The path is returned as a list of Plane objects.

Each plane is created using an origin vector and X and Y basis vectors.  The
time stamps and Z basis vectors in the trajectory file are ignored.
"""

# Load the Rhino API.
# import rhinoscriptsyntax as rs


# Make sure that the Python libraries also contained within this course package
# are on the load path.  This adds the parent folder to the load path, assuming that this
# script is still located with the rhinoscripts/ subfolder of the Python library tree.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import open3d as o3d
import time
import numpy as np

# Load the Optitrack CSV file parser module.
import optitrack.csv_reader as csv
from optitrack.geometry import *

# Find the path to the test data file located alongside the script.
# filename = os.path.join( os.path.abspath(os.path.dirname(__file__)), "sample_optitrack_take.csv")
# filename = "Take 2022-03-28 03.51.46 PM.csv"
filename = "groundTruth.csv"

# Read the file.
take = csv.Take().readCSV(filename)

# Print out some statistics
print("Found rigid bodies:", take.rigid_bodies.keys())

# Process the first rigid body into a set of planes.
bodies = take.rigid_bodies

# for now:
xaxis = [1,0,0]
yaxis = [0,1,0]

body_edges = [[0,1],[1,2],[2,3],[3,4],[3,5],[5,6],[6,7],[7,8],[3,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],
                [0,16],[16,17],[17,18],[18,20],[15,19]]

bones_pos = []
if len(bodies) > 0:
    for body in bodies:
        bones = take.rigid_bodies[body]
        bones_pos.append(bones.positions)
        # for pos,rot in zip(body.positions, body.rotations):
        #     if pos is not None and rot is not None:
        #         xaxis, yaxis = quaternion_to_xaxis_yaxis(rot)
                # plane = rs.PlaneFromFrame(pos, xaxis, yaxis)

                # # create a visible plane, assuming units are in meters
                # rs.AddPlaneSurface( plane, 0.1, 0.1 )
bones_pos = np.array(bones_pos).T.tolist()
colors = [[1, 0, 0] for i in range(len(body_edges))]
keypoints = o3d.geometry.PointCloud()
keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])
keypoints_center = keypoints.get_center()
keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])
skeleton_joints = o3d.geometry.LineSet()
skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
center_skel = skeleton_joints.get_center()
skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)
skeleton_joints.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()

vis.create_window()
# This plot the entire skeleton
vis.add_geometry(skeleton_joints)
vis.add_geometry(keypoints)

time.sleep(1)
for i in range(1000,2003):
    new_joints = bones_pos[i]
    print("NEW JOINTS ------- ",new_joints)
    # center_skel = skeleton_joints.get_center()

# File "/Users/letiziagirardi/Desktop/MotionCaptureLaboratory-main/python/demo/example_csv_loading.py", line 88, in <module>
# skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
# RuntimeError: Unable to cast Python instance to C++ type (compile in debug mode for details)
# ------------------------ ERROR i = 1370
# NEW JOINTS -------  [[-0.245972, 0.96495, -0.011748], [-0.242729, 1.042759, -0.011845], [-0.295498, 1.22621, -0.042028], [-0.276671, 1.437679, -0.022342], [-0.269548, 1.584029, -0.037125], [-0.279713, 1.401174, -0.064654], [-0.248889, 1.429697, -0.179419], [0.039565, 1.422109, -0.140807], [0.237982, 1.476942, -0.05328], [-0.283333, 1.394465, 0.010881], None, [0.015764, 1.426615, 0.10997], [0.197268, 1.479585, 0.0505], [-0.241895, 0.964662, -0.106632], [-0.227335, 0.513709, -0.08899], [-0.273246, 0.054937, -0.072116], [-0.250049, 0.965238, 0.083135], [-0.225068, 0.514409, 0.086417], [-0.278219, 0.05643, 0.103588], [-0.131537, -0.008423, -0.069103], [-0.136148, -0.006178, 0.104583]]

    # print("NEW JOINTS ------- ",new_joints)
    skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints.points = o3d.utility.Vector3dVector(new_joints)

    # This plot the entire skeleton
    vis.update_geometry(skeleton_joints)
    vis.update_geometry(keypoints)

    vis.update_renderer()
    vis.poll_events()

    time.sleep(0.5)

vis.run()
