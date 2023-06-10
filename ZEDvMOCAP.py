import numpy as np
import os
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json
import ZED.ZED_alignment as ZED
import MOCAP.MOCAP_alignment as MOCAP

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
        "Rclavicle": [2,7],"Rshoulder":[7,8], "Rarm":[8,9],"Rforearm":[9,10],
        "Lclavicle": [2,3],"Lshoulder":[3,4], "Larm":[4,5],"Lforearm":[5,6],
        "Rhip":[0,11], "Rthigh":[11,12],"Rshin":[12,13],
        "Lhip":[0,14], "Lthigh":[14,15],"Lshin":[15,16],
        "Rfoot":[13,17],"Lfoot":[16,18]}

MOCAP_mapping=[0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,4]

lower_body_indices=[0,11,12,13,14,15,16,17,18]

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
    ax1.view_init(azim=49, elev=8)
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

    for bone, indices in MOCAP_bones.items():
        idx1, idx2 = indices
        ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='green')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.3,0.3])
    ax2.set_ylim([-0.3,0.3])
    ax2.set_zlim([-0.3,0.3])
    ax2.view_init(azim=37, elev=8)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.25))
    plt.suptitle(title+str(pose))
    plt.savefig(f'ZEDvMOCAP/ZEDvMOCAP_{pose}.png')
    #plt.show()

def plot_lower_skeletons(skeleton1, skeleton2, skeleton3, skeleton4, title):

    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(x1, z1, y1, marker='o', label='Reference skeleton (lower part)', color='#0545e8')

    # for bone, indices in lower_bones.items():
    #     idx1, idx2 = indices
    #     ax1.plot([x1[idx1], x1[idx2]], [z1[idx1], z1[idx2]], [y1[idx1], y1[idx2]], color='#0545e8')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', label='Sample skeleton (lower part)', color='#e83205')

    # for bone, indices in lower_bones.items():
    #     idx1, idx2 = indices
    #     ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='#e83205')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.legend()

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', label='Aligned reference skeleton (lower part)', color='#0545e8')

    # for bone, indices in lower_bones.items():
    #     idx1, idx2 = indices
    #     ax2.plot([x3[idx1], x3[idx2]], [z3[idx1], z3[idx2]], [y3[idx1], y3[idx2]], color='#0545e8')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', label='Aligned sample skeleton (lower part)', color='#e83205')

    # for bone, indices in lower_bones.items():
    #     idx1, idx2 = indices
    #     ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='#e83205')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.set_xlim([-0.8,0.8])
    ax2.set_ylim([-0.8,0.8])
    ax2.set_zlim([-0.8,0.8])
    ax2.legend()

    plt.suptitle(title)
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

    # for bone, indices in ZED_bones.items():
    #     idx1, idx2 = indices
    #     ax.plot([x[idx1], x[idx2]], [z[idx1], z[idx2]], [y[idx1], y[idx2]], color='blue')

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

def MPJPE(skeleton1, skeleton2):
    difference=[]

    if skeleton1.shape == skeleton2.shape:
        for i,joint in enumerate(skeleton1):
            difference.append(np.linalg.norm(skeleton1[i] - skeleton2[i]))

    res=0
    for x in difference:
        res+=x

    return res/len(skeleton1)

def plot_pose(pose_state):

    colors = []
    for state in pose_state:
        if state == "T-POSE":
            colors.append('blue')
        elif state == "INTERMEDIATE":
            colors.append('orange')
        else:
            colors.append('gray')

    temp = [0] * len(colors)

    plt.scatter(range(len(colors)), temp, c=colors, label='Pose State')
    plt.xlabel('Timestamps')
    plt.ylabel('Pelvis vertical position')
    plt.title('Pose State over Time')

def main():

    mpjpe_array=[]
    lower_mpjpe_array=[]
    tot_disparity=0
    tot_lower_disparity=0
    tot_mpjpe_disparity=0
    tot_mpjpe_lower_disparity=0
    desired_bone_length=4.5

    if len(sys.argv) > 2:
        file_skeleton1 = sys.argv[1]
        file_skeleton2 = sys.argv[2]
        #print("Computing alignment between "+file_skeleton1+" at frame "+frame_skeleton1+" and "+file_skeleton2+" at frame "+frame_skeleton2)
    else:
        print("Not enough arguments")
        exit(1)

    ZED_keypositions=ZED.main(file_skeleton1)
    MOCAP_keypositions=MOCAP.main(file_skeleton2)

    keypositions=[]

    if len(ZED_keypositions)<len(MOCAP_keypositions):
        for i in range(len(ZED_keypositions)):
            temp=[ZED_keypositions[i],MOCAP_keypositions[i]]
            keypositions.append(temp)
    else:
        for i in range(len(MOCAP_keypositions)):
            temp=[ZED_keypositions[i],MOCAP_keypositions[i]]
            keypositions.append(temp)

    print(keypositions)

    for i,x in enumerate(keypositions):
        print(i)
        skeleton1 = read_skeletonZED(file_skeleton1, x[0])
        skeleton2 = read_skeletonMOCAP(file_skeleton2, x[1])

        ZED_skeleton=skeleton1[ZED_index]
        MOCAP_skeleton=skeleton2[MOCAP_mapping]

        bone_length1=0
        bone_length2=0

        for bone, indices in ZED_bones.items():
            idx1, idx2 = indices
            bone_length1+=compute_bone_length(ZED_skeleton[idx1],ZED_skeleton[idx2])

        for bone, indices in MOCAP_bones.items():
            idx1, idx2 = indices
            bone_length2+=compute_bone_length(MOCAP_skeleton[idx1],MOCAP_skeleton[idx2])

        # ZED_skeleton=scale_skeleton(ZED_skeleton, bone_length1,desired_bone_length)
        # MOCAP_skeleton=scale_skeleton(MOCAP_skeleton, bone_length2,desired_bone_length)

        ZED_skeleton = center_skeleton(ZED_skeleton)
        MOCAP_skeleton = center_skeleton(MOCAP_skeleton)

        # Reshape the arrays for Procrustes transformation
        ZED_skeleton_2d = ZED_skeleton.reshape(20, 3)
        MOCAP_skeleton_2d = MOCAP_skeleton.reshape(20, 3)

        # print(ZED_skeleton_2d)
        # print(MOCAP_skeleton_2d)

        # ZED_skeleton_2d_axis=  np.copy(ZED_skeleton_2d)
        # MOCAP_skeleton_2d_axis=  np.copy(MOCAP_skeleton_2d)
        #
        # ZED_skeleton_2d_axis[:][0]=ZED_skeleton_2d[:][2]
        # ZED_skeleton_2d_axis[:][2]=ZED_skeleton_2d[:][0]

        # ZED_skeleton_2d = ZED_skeleton_2d[:, [2, 1, 0]]
        # MOCAP_skeleton_2d = MOCAP_skeleton_2d[:, [0, 2, 1]]

        mtx1, mtx2, disparity = procrustes(MOCAP_skeleton_2d, ZED_skeleton_2d)
        aligned_skeleton1 = mtx1.reshape(20, 3)
        aligned_skeleton2 = mtx2.reshape(20, 3)

        plot_skeletons(ZED_skeleton,MOCAP_skeleton, mtx2, mtx1,i, "Aligned Skeletons (ZED v MOCAP)")

        mpjpe=MPJPE(aligned_skeleton1,aligned_skeleton2)
        print(mpjpe)
        mpjpe_array.append(mpjpe)

        tot_disparity+=disparity
        '''--------------------------------------------'''
        lower_body_ZED_skeleton=ZED_skeleton[lower_body_indices]
        lower_body_MOCAP_skeleton=MOCAP_skeleton[lower_body_indices]

        lower_body_ZED_skeleton = center_skeleton(lower_body_ZED_skeleton)
        lower_body_MOCAP_skeleton = center_skeleton(lower_body_MOCAP_skeleton)

        lower_body_ZED_skeleton_2d = lower_body_ZED_skeleton.reshape(9, 3)
        lower_body_MOCAP_skeleton_2d = lower_body_MOCAP_skeleton.reshape(9, 3)

        mtx1, mtx2, lower_disparity = procrustes(lower_body_ZED_skeleton_2d, lower_body_MOCAP_skeleton_2d)

        aligned_lower_body_skeleton1 = mtx1.reshape(9, 3)
        aligned_lower_body_skeleton2 = mtx2.reshape(9, 3)

        lower_mpjpe=MPJPE(aligned_lower_body_skeleton1,aligned_lower_body_skeleton2)
        print(lower_mpjpe)
        lower_mpjpe_array.append(mpjpe)

        #plot_lower_skeletons(lower_body_ZED_skeleton,lower_body_MOCAP_skeleton,aligned_lower_body_skeleton1,aligned_lower_body_skeleton2,"ZEDvMOCAP Aligned Lower body skeletons")

        tot_lower_disparity+=lower_disparity

    print("Total disparity:",tot_disparity/len(keypositions))

    #os.system(' ffmpeg -framerate 20 -i frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p ZEDvMOCAP.mp4 -y')

if __name__ == '__main__':
    main()
