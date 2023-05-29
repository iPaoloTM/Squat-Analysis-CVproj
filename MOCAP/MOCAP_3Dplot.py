import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import json
import sys

def main():
    if len(sys.argv) > 2:
        file_to_read = sys.argv[1]
        frame_number = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    with open('../body_data/second_attempt/'+file_to_read+'MOCAP.json', 'r') as f:
        data = json.load(f)

    frame_data = data[int(frame_number)]
    body_data = frame_data

    # keypoints = np.array(keypoints)
    # print(keypoints)
    keypoints=[]
    for joint in body_data['keypoints']:
        keypoints.append(joint['Position'])

    keypoints = np.array(keypoints)

    print(keypoints)

    # split the points into x, y, z coordinates
    x = [p[0] for p in keypoints]
    y = [p[1] for p in keypoints]
    z = [p[2] for p in keypoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, z, y, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0, 1.8])
    #ax.view_init(azim=-90, elev=90)
    ax.invert_yaxis()

    plt.show()

if __name__ == '__main__':
    main()
