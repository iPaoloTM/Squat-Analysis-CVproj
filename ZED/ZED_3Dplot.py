from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys

def read_skeleton(file_name, frame):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    keypoints = []
    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                keypoints.append(body_part['keypoint'])

    return np.array(keypoints[0])

def plot_skeleton(skeleton):

    x = [point[0] for point in skeleton] #keypoints[0]]
    y = [point[1] for point in skeleton]
    z = [point[2] for point in skeleton]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-1.5, 0.8])
    ax.set_zlim([-0.5, 0.5])
    ax.view_init(azim=-90, elev=90)

    plt.show()

import numpy as np

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

def main():

    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    keypoints = read_skeleton(file_name,frame)
    print(keypoints)
    keypoints = center_skeleton(keypoints)
    print(keypoints)

    plot_skeleton(keypoints)

if __name__ == '__main__':
    main()
