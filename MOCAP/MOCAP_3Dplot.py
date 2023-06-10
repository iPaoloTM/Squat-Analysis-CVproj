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

    ax.plot([skeleton[2][0], skeleton[0][0]], [skeleton[2][2], skeleton[0][2]], [skeleton[2][1], skeleton[0][1]],  'r--', label='Back Line')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.view_init(azim=-118, elev=8)
    ax.invert_yaxis()
    plt.title("MOCAP skeleton")
    plt.show()

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

def compute_angle(x1,y1,x2,y2):

    # x1=0.09
    # y1=0.06
    # x2=0.5
    # y2=0.5

    print(x1)
    print(y1)
    print(x2)
    print(y2)

    if (x2 - x1)!=0:
        slope1 = (y2 - y1) / (x2 - x1)
    elif (x2-x1)==0:
        slope1=math.inf

    angle1 = math.degrees(math.atan(slope1))

    angle_diff = abs(90 - angle1)

    return angle_diff

def compute_bone_length(joint1, joint2):
    """
    Calculate the Euclidean distance between two joints (bone length).
    """
    return np.linalg.norm(joint2 - joint1)

def scale_skeleton(skeleton, total_bone_length, desired_bone_length):
    """
    Compute the scaling factor given the total bone length and the desired bone length and
    scale the skeleton by applying the scaling factor to each bone length.
    """
    scaling_factor=desired_bone_length / total_bone_length
    scaled_skeleton = skeleton * scaling_factor
    return scaled_skeleton

def main():

    desired_bone_length=4.5

    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    skeleton = read_skeleton(file_name,frame)
    bone_length=0
    for bone, indices in bones.items():
        idx1, idx2 = indices
        bone_length+=compute_bone_length(skeleton[idx1],skeleton[idx2])
    skeleton=scale_skeleton(skeleton, bone_length,desired_bone_length)
    skeleton = center_skeleton(skeleton)

    theta = compute_angle(skeleton[0][0],skeleton[0][1],skeleton[2][0],skeleton[2][1])

    print("Back angle:",theta)

    plot_skeleton(skeleton)

if __name__ == '__main__':
    main()
