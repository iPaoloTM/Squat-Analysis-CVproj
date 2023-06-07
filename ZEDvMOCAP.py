import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

ZED_bones={"pelvis+abs": [0,1], "chest": [1,2], "neck": [2,19],
       "Rclavicle":[2,7],"Rshoulder":[7,8],"Rarm":[8,9], "Rforearm":[9,10],
       "Lclavicle":[2,3],"Lshoulder":[3,4], "Larm":[4,5], "Lforearm":[5,6],
       "Rhip":[0,11], "Rthigh":[11,12],"Rshin":[12,13],
       "Lhip":[0,14], "Lthigh":[14,15],"Lshin":[15,16],
       "Rfoot":[13,17],"Lfoot":[16,18]}

ZED_index=[0,1,3,4,5,6,7,11,12,13,14,22,23,24,18,19,20,25,21,26]

# MOCAP_bones={"pelvis": [0,1], "chest": [1,2], "neck": [2,3],
#        "Rclavicle":[2,4],"Rshoulder":[4,5],"Rarm":[5,6], "Rforearm":[6,7],
#        "Lclavicle":[2,8],"Lshoulder":[8,9], "Larm":[9,10], "Lforearm":[10,11],
#        "Rhip":[0,12], "Rthigh":[12,13],"Rshin":[13,14],
#        "Lhip":[0,15], "Lthigh":[15,16],"Lshin":[16,17],
#        "Rfoot":[14,18],"Lfoot":[17,19]}

MOCAP_bones={"pelvis": [0,1], "chest": [1,2], "neck": [2,19],
        "Rclavicle": [2,3],"Rshoulder":[3,4],"Rarm":[4,5],"Rforearm":[5,6],
        "Rclavicle": [2,7],"Lshoulder":[7,8], "Larm":[8,9],"Lforearm":[9,10],
        "Rhip":[0,11], "Rthigh":[11,12],"Rshin":[12,13],
        "Lhip":[0,14], "Lthigh":[14,15],"Lshin":[15,16],
        "Rfoot":[13,17],"Lfoot":[16,18]}
MOCAP_mapping=[0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,4]

def plot_skeletons(skeleton1, skeleton2, skeleton3, skeleton4, pose, title):

    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(z1, x1, y1, marker='o', color='orange', label='ZED')

    for bone, indices in ZED_bones.items():
        idx1, idx2 = indices
        ax1.plot([z1[idx1], z1[idx2]], [x1[idx1], x1[idx2]], [y1[idx1], y1[idx2]], color='orange')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', color='green', label='MOCAP')

    for bone, indices in MOCAP_bones.items():
        idx1, idx2 = indices
        ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='green')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.legend()
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.5))

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', color='orange', label='ZED')

    for bone, indices in ZED_bones.items():
        idx1, idx2 = indices
        ax2.plot([x3[idx1], x3[idx2]], [z3[idx1], z3[idx2]], [y3[idx1], y3[idx2]], color='orange')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', color='green', label='MOCAP')

    for bone, indices in ZED_bones.items():
        idx1, idx2 = indices
        ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='green')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_zlim([-0.3, 0.3])
    ax2.legend()
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.25))
    plt.suptitle(title+str(pose))
    plt.savefig(f'frame_{pose}.png')
    plt.show()

def read_skeletonZED(file_name, frame):
    skeleton=[]
    # Load the second JSON file
    with open('./body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                skeleton.append(body_part['keypoint'])

    return np.array(skeleton[0])

def read_skeletonMOCAP(file_name, frame):
    with open('./body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    frame_data = data[int(frame)]
    body_data = frame_data

    keypoints=[]
    for joint in body_data['keypoints']:
        keypoints.append(joint['Position'])

    return np.array(keypoints)

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

def plot_skeleton(skeleton):

    x = [point[0] for point in skeleton] #keypoints[0]]
    y = [point[1] for point in skeleton]
    z = [point[2] for point in skeleton]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(x, z, y, marker='o')

    for bone, indices in ZED_bones.items():
        idx1, idx2 = indices
        ax.plot([x[idx1], x[idx2]], [z[idx1], z[idx2]], [y[idx1], y[idx2]], color='blue')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.view_init(azim=-90, elev=90)
    plt.show()


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

    tot_disparity=0
    desired_bone_length=4.5

    if len(sys.argv) > 2:
        file_skeleton1 = sys.argv[1]
        file_skeleton2 = sys.argv[2]
        #print("Computing alignment between "+file_skeleton1+" at frame "+frame_skeleton1+" and "+file_skeleton2+" at frame "+frame_skeleton2)
    else:
        print("Not enough arguments")
        exit(1)

    keypositions=[[272,542], [443,1501], [615,2461], [787,3421], [797,3516], [806,3601], [827,3873], [864,4000], [879,4080], [901,4174], [1140,6090], [1159,6229], [1174,6338], [1204,6719], [1246,6849], [1261,6919], [1288,7008], [1515,8893], [1529,9003], [1546,9106], [1592,9435], [1614,9600], [1631,9694], [1658,9803]]

    for x in keypositions:

        skeleton1 = read_skeletonZED(file_skeleton1, x[0])
        skeleton1 = center_skeleton(skeleton1)
        skeleton2 = read_skeletonMOCAP(file_skeleton2, x[1])
        skeleton2 = center_skeleton(skeleton2)

        skeleton_new_ZED=skeleton1[ZED_index]
        skeleton_new_MOCAP=skeleton2[MOCAP_mapping]

        bone_length1=0
        bone_length2=0

        print('ZED')
        for bone, indices in ZED_bones.items():
            idx1, idx2 = indices
            bone_length1+=compute_bone_length(skeleton_new_ZED[idx1],skeleton_new_ZED[idx2])
        print('mocap')

        for bone, indices in MOCAP_bones.items():
            idx1, idx2 = indices
            bone_length2+=compute_bone_length(skeleton_new_MOCAP[idx1],skeleton_new_MOCAP[idx2])
        print(bone_length1)
        print(bone_length2)


        skeleton_new_ZED=scale_skeleton(skeleton_new_ZED, bone_length1,desired_bone_length)
        skeleton_new_MOCAP=scale_skeleton(skeleton_new_MOCAP, bone_length2,desired_bone_length)

        # Reshape the arrays for Procrustes transformation
        skeleton1_2d = skeleton_new_ZED.reshape(20, 3)
        skeleton2_2d = skeleton_new_MOCAP.reshape(20, 3)

        mtx1, mtx2, disparity = procrustes(skeleton1_2d, skeleton2_2d)
        aligned_skeleton1 = mtx1.reshape(20, 3)
        aligned_skeleton2 = mtx2.reshape(20, 3)
        pose=0
        plot_skeletons(skeleton_new_ZED,skeleton_new_MOCAP,aligned_skeleton1,aligned_skeleton2, x, "Aligned Skeletons (ZED v MOCAP)")

        print("General Disparity:",disparity)

if __name__ == '__main__':
    main()
