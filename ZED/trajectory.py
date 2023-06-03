import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm

# load the JSON file
with open('../body_data/Squat2.json', 'r') as f:
    data = json.load(f)

# Extract the "keypoint" vectors
keypoints = []
for body in data.values():
    for body_part in body['body_list']:
        keypoints.append(body_part['keypoint'])
keypoints = np.array(keypoints)

# get the coordinates of the specific point (e.g. nose)
point_idx = 1
point_coords = keypoints[:,point_idx,:] #(3164,18,2)

valid_points = []
for i in range(len(point_coords)):
    #print(point_coords[i])
    if not np.array_equal(point_coords[i], [-1.,-1.]): #[np.nan, np.nan, np.nan]):
        valid_points.append(point_coords[i])

valid_points = np.array(valid_points)

y_values = valid_points[:, 1] # y values for coloring

# normalize y_values to [0,1] for use with colormap
y_values_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))

# define colormap
cmap = cm.get_cmap('RdBu_r')

# Plot trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valid_points[:,0], valid_points[:,1], valid_points[:,2],c=cmap(y_values_norm), alpha=0.1)
ax.invert_yaxis()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(azim=-90, elev=90)
ax.set_xlim([-0.8, 0.8])
ax.set_ylim([-0.8, 0.8])
ax.set_zlim([-5, -3])
plt.show()
