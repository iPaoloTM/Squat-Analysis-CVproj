from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys

def read_skeleton(file_name, frame):
    with open('../body_data/first_attempt/'+file_name+'.json', 'r') as f:
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
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-2, -0.5])
    ax.set_zlim([-4.5, -3.5])
    ax.view_init(azim=-90, elev=90)

    plt.show()

def main():

    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    keypoints = read_skeleton(file_name,frame)
    print(keypoints)

    plot_skeleton(keypoints)

if __name__ == '__main__':
    main()
