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

def read_skeleton(file_name):
        with open('../body_data/'+file_name+'.json', 'r') as f:
            data = json.load(f)

        keypoints = [[]]
        for i,frame in enumerate(data):
            keypoints.append([])
            for joint in frame['keypoints']:
                keypoints[i].append(joint['Position'])

        return np.array(keypoints)


def plot_skeleton(skeleton):

    #print(skeleton.shape)
    #print(skeleton)

    # split the points into x, y, z coordinates
    x = [p[0] for p in skeleton]
    y = [p[1] for p in skeleton]
    z = [p[2] for p in skeleton]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, z, y, marker='o', color='green')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax.plot([x[idx1], x[idx2]], [z[idx1], z[idx2]], [y[idx1], y[idx2]], color='green')

    ax.plot([skeleton[0][0], skeleton[0][0]], [skeleton[0][1], skeleton[0][1]], [skeleton[0][2]-1, skeleton[0][2]+1], 'r--', label='Vertical Line')

    vline_x = skeleton[3][0]
    vline_y = skeleton[3][1]
    vline_z = skeleton[3][2]

    ax.plot([skeleton[2][0], skeleton[1][0]], [skeleton[2][2], skeleton[1][2]], [skeleton[2][1], skeleton[1][1]],  'r--', label='Vertical Line')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(azim=-118, elev=8)
    ax.invert_yaxis()
    plt.title("MOCAP skeleton")
    plt.show()

def center_skeleton(skeleton):

    r_foot_15 = skeleton[15] # 15, 19
    l_foot_18 = skeleton[18] # 18 20

    x_baricentro = (r_foot_15[0] + l_foot_18[0] ) / 2
    y_baricentro = (r_foot_15[1] + l_foot_18[1] ) / 2
    z_baricentro = (r_foot_15[2]  + l_foot_18[2] ) / 2

    zero_in_foots = [x_baricentro, y_baricentro, z_baricentro]
    # Compute the displacement vector
    displacement_vector = -np.array(zero_in_foots)

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

def compute_angle(x1,y1,x2,y2):

    if (x2 - x1)!=0:
        slope1 = (y2 - y1) / (x2 - x1)
    elif (x2-x1)==0:
        slope1=math.inf

    angle1 = math.degrees(math.atan(slope1))

    return angle1

def main():
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        print("Not enough arguments")
        exit(1)

    keypoints = read_skeleton(file_name)

    keypoints = center_skeleton(keypoints)

    print("Back angle:",compute_angle(keypoints[2][2],keypoints[2][0], keypoints[1][2], keypoints[1][0]))

    plot_skeleton(keypoints)



if __name__ == '__main__':
    main()
