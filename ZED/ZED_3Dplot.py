from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys
import math

bones={"pelvis+abs": [0,1], "chest": [1,2], "neck": [3,26],
       "Rclavicle":[3,11],"Rshoulder":[11,12],"Rarm":[12,13], "Rforearm":[13,14],
       "chest1":[2,11],"chest2":[2,3],"chest3":[2,4],
       "Lclavicle":[3,4],"Lshoulder":[4,5], "Larm":[5,6], "Lforearm":[6,7],
       "Rhip":[0,22], "Rthigh":[22,23],"Rshin":[23,24],
       "Lhip":[0,18], "Lthigh":[18,19],"Lshin":[19,20],
       "Rfoot":[25,33],"Rankle":[24,33],"Lfoot":[21,32],"Lankle":[20,32]}

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


    ax.scatter(x, z, y, marker='o', color='orange', label='ZED skeleton')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax.plot([x[idx1], x[idx2]], [z[idx1], z[idx2]], [y[idx1], y[idx2]], color='orange')

    ax.plot([skeleton[0][0], skeleton[0][0]], [skeleton[0][1], skeleton[0][1]], [skeleton[0][2]+1,skeleton[0][2]-1], 'r--', label='Vertical Line')
    ax.plot([skeleton[0][0], skeleton[1][0]], [skeleton[0][2], skeleton[1][2]], [skeleton[0][1],skeleton[1][1]], 'r--', label='Back line Line')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.view_init(azim=148, elev=7)
    plt.title("ZED skeleton")
    plt.show()

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

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

def compute_angle(x1,y1,x2,y2,x3,y3,x4,y4):

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

    desired_bone_length=4.5

    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    skeleton = read_skeleton(file_name,frame)
    print(skeleton)
    bone_length=0
    for bone, indices in bones.items():
        idx1, idx2 = indices
        bone_length+=compute_bone_length(skeleton[idx1],skeleton[idx2])
    skeleton=scale_skeleton(skeleton, bone_length,desired_bone_length)
    skeleton = center_skeleton(skeleton)
    print(skeleton)

    print("Back angle:",compute_angle(skeleton[0][0],skeleton[0][1], skeleton[1][0], skeleton[1][1],skeleton[0][0],skeleton[0][1],skeleton[0][0],skeleton[0][1]+0.5))

    plot_skeleton(skeleton)


if __name__ == '__main__':
    main()
