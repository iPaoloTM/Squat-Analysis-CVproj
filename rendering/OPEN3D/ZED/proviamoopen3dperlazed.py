import numpy as np
import open3d as o3d

print("Let's draw a box using o3d.geometry.LineSet.")
# json from zed
points = [

]
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


# filename = "Takes/gt1_MOCAP.csv"
# take = csv.Take().readCSV(filename)

# body = take.rigid_bodies.copy()

body_edges = [[0,1],[1,2],[2,3],[3,4],[3,5],[5,6],[6,7],[7,8],[3,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],
                [0,16],[16,17],[17,18],[18,20],[15,19]]

LFHAND = 8
RHAND = 12
HIP = 0

interpolation = ['Linear interpolation', 'Linear interpolation+Kalman Filter', 'Kalman predictor', 'No interpolation']
while True:
    print('What kind of interpolation do you want to use?:')
    for i, inter in enumerate(interpolation):
        print(i, '-', inter)
    choice = input()
    if choice.isnumeric():
        choice = int(choice)
        if 0 <= choice < len(interpolation):
            interpolation = interpolation[choice]
            break

print(interpolation)

bones_pos = np.array(bones_pos).transpose(1,0,2).tolist()

bone_joint = bones_pos[0]

# Generation of the skeleton
colors = [[1, 0, 0] for i in range(len(body_edges))]

keypoints = o3d.geometry.PointCloud()

keypoints.points = o3d.utility.Vector3dVector(bone_joint)
keypoints_center = keypoints.get_center()
keypoints.points = o3d.utility.Vector3dVector(bone_joint)

skeleton_joints = o3d.geometry.LineSet()
skeleton_joints.points = o3d.utility.Vector3dVector(bone_joint)
center_skel = skeleton_joints.get_center()

body_trajectory = o3d.geometry.LineSet()
skeleton_joints.points = o3d.utility.Vector3dVector(bone_joint)
skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)
skeleton_joints.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()

WINDOW_WIDTH=1920
WINDOW_HEIGHT=1080

# Insertion of geometries in the visualizer
vis.create_window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
vis.add_geometry(skeleton_joints)
vis.add_geometry(keypoints)
vis.add_geometry(body_trajectory)


for i in range(len(bones_pos)):
    # If the measurements are correct the model updates
    if None not in bones_pos[i]:
        new_joints = bones_pos[i]
    else:
        missed_body += 1

    left_hand = new_joints[LFHAND]
    right_hand = new_joints[RHAND]

    skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints.points = o3d.utility.Vector3dVector(new_joints)

    # Update of skeleton
    vis.update_geometry(skeleton_joints)
    vis.update_geometry(keypoints)
    vis.update_geometry(body_trajectory)

    vis.update_renderer()
    vis.poll_events()

vis.run()
#
# colors = [[1, 0, 0] for _ in range(len(lines))]
# line_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(points),
#     lines=o3d.utility.Vector2iVector(lines),
# )
# line_set.colors = o3d.utility.Vector3dVector(colors)
#
# lookat = np.array([0.5, 0.5, 0.5])  # Camera center
# up = np.array([0, -1, 0])           # Up direction
# front = np.array([0, 0, -1])        # Front direction
# zoom = 0.8
#
# o3d.visualization.draw_geometries([line_set])
