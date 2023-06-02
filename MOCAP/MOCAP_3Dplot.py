import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import json
import sys

def read_skeleton(file_name, frame):
    with open('../body_data/second_attempt/'+file_name+'MOCAP.json', 'r') as f:
        data = json.load(f)

    frame_data = data[int(frame)]
    body_data = frame_data

    keypoints=[]
    for joint in body_data['keypoints']:
        keypoints.append(joint['Position'])

    return np.array(keypoints)

def plot_skeleton(skeleton):

    # split the points into x, y, z coordinates
    x = [p[0] for p in skeleton]
    y = [p[1] for p in skeleton]
    z = [p[2] for p in skeleton]

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

def main():
    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    keypoints = read_skeleton(file_name,frame)

    plot_skeleton(keypoints)

if __name__ == '__main__':
    main()
