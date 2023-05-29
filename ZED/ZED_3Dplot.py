from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys

if len(sys.argv) > 2:
    file_to_read = sys.argv[1]
    frame_number = sys.argv[2]
else:
    print("Not enough arguments")
    exit(1)

with open('../body_data/first_attempt/'+file_to_read+'.json', 'r') as f:
    data = json.load(f)

keypoints = []
for i,body in enumerate(data.values()):
    if i==int(frame_number):
        for body_part in body['body_list']:
            keypoints.append(body_part['keypoint'])

keypoints = np.array(keypoints)

print(keypoints[0])

x = [point[0] for point in keypoints[0]]
y = [point[1] for point in keypoints[0]]
z = [point[2] for point in keypoints[0]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([-1, 1])
ax.set_ylim([-2, 0])
ax.set_zlim([-4.5, -3.5])
ax.view_init(azim=-90, elev=90)

plt.show()
