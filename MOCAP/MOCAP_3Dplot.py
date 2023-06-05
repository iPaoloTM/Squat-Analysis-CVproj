import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import json
import sys
import math

bones={"pelvis": [0,1], "abs": [1,2], "chest": [2,3], "neck": [3,4],
       "Rclavicle":[3,5],"Rshoulder":[5,6],"Rarm":[6,7], "Rforearm":[7,8],
       "Lclavicle":[3,9],"Lshoulder":[9,10], "Larm":[10,11], "Lforearm":[11,12],
       "Rhip":[0,13], "Rthigh":[13,14],"Rshin":[14,15],
       "Lhip":[0,16], "Lthigh":[16,17],"Lshin":[17,18],
       "Rfoot":[15,19],"Lfoot":[18,20]}

def read_skeleton(file_name, frame):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    frame_data = data[int(frame)]
    body_data = frame_data

    keypoints=[]
    for joint in body_data['keypoints']:
        keypoints.append(joint['Position'])

    return np.array(keypoints)

def plot_skeleton(skeleton):

    print(skeleton.shape)
    print(skeleton)

    # split the points into x, y, z coordinates
    x = [p[0] for p in skeleton]
    y = [p[1] for p in skeleton]
    z = [p[2] for p in skeleton]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, z, y, marker='o')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax.plot([x[idx1], x[idx2]], [z[idx1], z[idx2]], [y[idx1], y[idx2]], color='blue')

    vline_x = skeleton[0][0]
    vline_y = skeleton[0][1]
    vline_z = max(skeleton[0][2], skeleton[1][2]) + 1  # Extend the line above the highest point

    ax.plot([vline_x, vline_x], [vline_y, vline_y], [vline_z, min(skeleton[0][2], skeleton[1][2])], 'r--', label='Vertical Line')

    ax.scatter(skeleton[0][0],skeleton[0][0],skeleton[0][2]+0.6)
    ax.scatter(skeleton[2][0],skeleton[2][2], skeleton[2][1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.4, 1])
    ax.set_ylim([-0.4, 1])
    ax.set_zlim([-1.8, 0.8])
    #ax.view_init(azim=-90, elev=90)
    ax.invert_yaxis()

    plt.show()

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

def compute_angle(x1,y1,x2,y2,x3,y3,x4,y4):

    print(x1)
    print(y1)
    print(x2)
    print(y2)
    print(x3)
    print(y3)
    print(x4)
    print(y4)

    if (((x2 - x1)!=0) & ((x4 - x3)!=0)):
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)
    elif (((x2 - x1)==0) & ((x4 - x3)==0)):
        slope1=math.inf
        slope2=math.inf
    elif ((x2-x1)==0):
        slope1=math.inf
        slope2 = (y4 - y3) / (x4 - x3)
    elif ((x4 - x3)==0):
        slope1 = (y2 - y1) / (x2 - x1)
        slope2=math.inf

    angle1 = math.degrees(math.atan(slope1))
    angle2 = math.degrees(math.atan(slope2))

    angle_diff = abs(angle2 - angle1)

    return angle_diff

def main():
    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    keypoints = read_skeleton(file_name,frame)

    keypoints = center_skeleton(keypoints)

    plot_skeleton(keypoints)

    print("Back angle:",compute_angle(keypoints[0][1],keypoints[0][2], keypoints[1][1], keypoints[1][2],keypoints[0][1],keypoints[0][2],keypoints[0][1],keypoints[0][2]+0.6))


if __name__ == '__main__':
    main()
