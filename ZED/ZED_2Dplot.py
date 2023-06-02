import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import sys
import json

def read_skeleton(file_name, frame):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    keypoints = []
    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                keypoints.append(body_part['keypoint_2d'])

    return np.array(keypoints[0])

def plot_skeleton(skeleton):

    x = [point[0] for point in skeleton] #keypoints[0]]
    y = [point[1] for point in skeleton]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y,marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    ax.invert_yaxis()

    plt.show()

def main():

    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = int(sys.argv[2])
    else:
        print("No file name provided.")
        exit(1)

    keypoints=read_skeleton(file_name,frame)

    plot_skeleton(keypoints)

# ax.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], 'b-') #right shoulder
# ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], 'b-') #right arm
# ax.plot([points[3][0], points[4][0]], [points[3][1], points[4][1]], 'b-') #right forearm
# ax.plot([points[1][0], points[5][0]], [points[1][1], points[5][1]], 'b-') #left shoulder
# ax.plot([points[5][0], points[6][0]], [points[5][1], points[6][1]], 'b-') #left arm
# ax.plot([points[6][0], points[7][0]], [points[6][1], points[7][1]], 'b-') #left forearm
# ax.plot([points[1][0], points[8][0]], [points[1][1], points[8][1]], 'b-') #right part of back
# ax.plot([points[8][0], points[9][0]], [points[8][1], points[9][1]], 'b-') #right leg
# ax.plot([points[9][0], points[10][0]], [points[9][1], points[10][1]], 'b-') #right calf
# ax.plot([points[1][0], points[11][0]], [points[1][1], points[11][1]], 'b-') #left part of back
# ax.plot([points[11][0], points[12][0]], [points[11][1], points[12][1]], 'b-') #left leg
# ax.plot([points[12][0], points[13][0]], [points[12][1], points[13][1]], 'b-') #left calf
# ax.plot([points[16][0], points[17][0]], [points[16][1], points[17][1]], 'b-') #eye line

if __name__ == '__main__':
    main()
