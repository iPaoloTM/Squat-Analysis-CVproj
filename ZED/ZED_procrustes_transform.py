import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json
import ZED_alignment as ZED
import math

bones={"pelvis+abs": [0,1], "chest": [1,2], "neck": [3,26],
       "Rclavicle":[3,11],"Rshoulder":[11,12],"Rarm":[12,13], "Rforearm":[13,14],
       "Lclavicle":[3,4],"Lshoulder":[4,5], "Larm":[5,6], "Lforearm":[6,7],
       "chest1":[2,11],"chest2":[2,3],"chest3":[2,4],
       "Rhip":[0,22], "Rthigh":[22,23],"Rshin":[23,24],
       "Lhip":[0,18], "Lthigh":[18,19],"Lshin":[19,20],
       "Rfoot":[25,33],"Rankle":[24,33],"Lfoot":[21,32],"Lankle":[20,32]}

lower_bones={"Rhip":[0,1],"Rthigh":[1,2],"Rshin":[2,3],"Rankle":[3,4],"Rfoot1":[4,9],"Rfoot": [3,9],
       "Lhip":[0,5],"Lthigh":[5,6],"Lshin":[6,7],"Lankle":[7,8],"Lfoot1":[8,10],"Lfoot":[7,10]   }

lower_body_indices = [0, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33]

def plot_skeletons(skeleton1, skeleton2, skeleton3, skeleton4, pose,title):

    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(x1, z1, y1, marker='o', label='Reference skeleton', color='#0545e8')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax1.plot([x1[idx1], x1[idx2]], [z1[idx1], z1[idx2]], [y1[idx1], y1[idx2]], color='#0545e8')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', label='Sample skeleton', color='#e83205')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='#e83205')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.view_init(azim=151, elev=6)
    #ax1.legend()

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', label='Aligned reference skeleton', color='#0545e8')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax2.plot([x3[idx1], x3[idx2]], [z3[idx1], z3[idx2]], [y3[idx1], y3[idx2]], color='#0545e8')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', label='Aligned sample skeleton', color='#e83205')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='#e83205')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_zlim([-0.3, 0.3])
    ax2.view_init(azim=140, elev=5)
    #plt.savefig(f'ZED_reference_sample/ZED_reference_sample_{pose}.png')
    plt.suptitle(title)
    plt.show()

def plot_lower_skeletons(skeleton1, skeleton2, skeleton3, skeleton4, pose, title):

    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(x1, z1, y1, marker='o', label='Reference skeleton (lower part)', color='#0545e8')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax1.plot([x1[idx1], x1[idx2]], [z1[idx1], z1[idx2]], [y1[idx1], y1[idx2]], color='#0545e8')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', label='Sample skeleton (lower part)', color='#e83205')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='#e83205')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.view_init(azim=151, elev=6)

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', label='Aligned reference skeleton (lower part)', color='#0545e8')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax2.plot([x3[idx1], x3[idx2]], [z3[idx1], z3[idx2]], [y3[idx1], y3[idx2]], color='#0545e8')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', label='Aligned sample skeleton (lower part)', color='#e83205')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='#e83205')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.set_xlim([-0.8,0.8])
    ax2.set_ylim([-0.8,0.8])
    ax2.set_zlim([-0.3,1.2])
    ax2.view_init(azim=144, elev=6)

    plt.suptitle(title)
    #plt.savefig(f'ZED_lower_reference_sample/ZED_lower_reference_sample_{pose}.png')
    plt.show()

def read_skeleton(file_name, frame):

    skeleton=[]
    # Load the second JSON file
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                skeleton.append(body_part['keypoint'])

    return np.array(skeleton[0])

def center_skeleton(skeleton):
    """
    Align the skeleton's pelvis to the world center [0,0,0]
    """

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

def MPJPE(skeleton1, skeleton2):
    difference=[]

    if skeleton1.shape == skeleton2.shape:
        for i,joint in enumerate(skeleton1):
            difference.append(np.linalg.norm(skeleton1[i] - skeleton2[i]))

    res=0
    for x in difference:
        res+=x

    return res/len(skeleton1)

def compute_angle(x1,y1,x2,y2):

    if (x2 - x1)!=0:
        slope1 = (y2 - y1) / (x2 - x1)
    elif (x2-x1)==0:
        slope1=math.inf

    angle1 = math.degrees(math.atan(slope1))

    angle_diff = abs(90 - angle1)

    return angle_diff

def main():

    desired_bone_length=4.5

    if len(sys.argv) > 2:
        file_skeleton1 = sys.argv[1]
        #frame_skeleton1 = sys.argv[2]
        file_skeleton2 = sys.argv[2]
        #frame_skeleton2 = sys.argv[4]
        #print("Computing alignment between "+file_skeleton1+" at frame "+frame_skeleton1+" and "+file_skeleton2+" at frame "+frame_skeleton2)
    else:
        print("Not enough arguments")
        exit(1)

    keypositions1=ZED.main(file_skeleton1)
    keypositions2=ZED.main(file_skeleton2)

    keypositions=[]

    if len(keypositions1)<len(keypositions2):
        for i in range(len(keypositions1)):
            temp=[keypositions1[i],keypositions2[i]]
            keypositions.append(temp)
    else:
        for i in range(len(keypositions2)):
            temp=[keypositions1[i],keypositions2[i]]
            keypositions.append(temp)

    print(keypositions)

    tot_disparityP=0
    tot_disparityM=0
    tot_lower_disparity=0
    tot_lower_disparityM=0
    tot_theta_diff=0

    for i,x in enumerate(keypositions):
        print(i)
        skeleton1 = read_skeleton(file_skeleton1, x[0])
        skeleton2 = read_skeleton(file_skeleton2, x[1])
        bone_length1=0
        bone_length2=0
        for bone, indices in bones.items():
            idx1, idx2 = indices
            bone_length1+=compute_bone_length(skeleton1[idx1],skeleton1[idx2])
            bone_length2+=compute_bone_length(skeleton2[idx1],skeleton2[idx2])
        skeleton1=scale_skeleton(skeleton1, bone_length1,desired_bone_length)
        skeleton2=scale_skeleton(skeleton2, bone_length2,desired_bone_length)
        skeleton1 = center_skeleton(skeleton1)
        skeleton2 = center_skeleton(skeleton2)

        # Padding the smaller skeleton with zeros to match the shape of the larger skeleton
        max_points = max(skeleton1.shape[0], skeleton2.shape[0])
        if skeleton1.shape[0] < max_points:
            skeleton1 = np.pad(skeleton1, ((0, max_points - skeleton1.shape[0]), (0, 0)), mode='constant')
        elif skeleton2.shape[0] < max_points:
            skeleton2 = np.pad(skeleton2, ((0, max_points - skeleton2.shape[0]), (0, 0)), mode='constant')

        #print("Skeleton1: ",skeleton1)
        #print("Skeleton2: ",skeleton2)

        #plot_skeletons(skeleton1,skeleton2,"Original Skeletons")

        # Reshape the arrays for Procrustes transformation
        skeleton1_2d = skeleton1.reshape(34, 3)
        skeleton2_2d = skeleton2.reshape(34, 3)

        mtx1, mtx2, disparity = procrustes(skeleton1_2d, skeleton2_2d)
        aligned_skeleton1 = mtx1.reshape(34, 3)
        aligned_skeleton2 = mtx2.reshape(34, 3)

        #plot_skeletons(skeleton1,skeleton2,aligned_skeleton1,aligned_skeleton2,i,"ZED Aligned Skeletons")
        mpjpe=MPJPE(aligned_skeleton1,aligned_skeleton2)
        #print("Procrustes disparity:",str(disparity))
        #print("MPJPE disparity:",str(mpjpe))
        tot_disparityP+=disparity
        tot_disparityM+=mpjpe

        lower_body_skeleton1=skeleton1[lower_body_indices]
        lower_body_skeleton2=skeleton2[lower_body_indices]

        lower_body_skeleton1 = center_skeleton(lower_body_skeleton1)
        lower_body_skeleton2 = center_skeleton(lower_body_skeleton2)

        lower_body_skeleton1_2d = lower_body_skeleton1.reshape(11, 3)
        lower_body_skeleton2_2d = lower_body_skeleton2.reshape(11, 3)

        mtx1, mtx2, lower_disparity = procrustes(lower_body_skeleton1_2d, lower_body_skeleton2_2d)

        aligned_lower_body_skeleton1 = mtx1.reshape(11, 3)
        aligned_lower_body_skeleton2 = mtx2.reshape(11, 3)

        #plot_lower_skeletons(lower_body_skeleton1,lower_body_skeleton2,aligned_lower_body_skeleton1,aligned_lower_body_skeleton2,i,"ZED Aligned Lower body skeletons")
        lower_mpjpe=MPJPE(aligned_lower_body_skeleton1,aligned_lower_body_skeleton2)
        #print("lowerbody procrustes disparity:",str(lower_disparity))
        #print("LowerMPJPE:",str(lower_mpjpe))
        tot_lower_disparityM+=lower_mpjpe
        tot_lower_disparity+=lower_disparity

        if i<21:
            theta1=compute_angle(skeleton1[0][2],skeleton1[0][1], skeleton1[2][2], skeleton1[2][1])
            theta2=compute_angle(skeleton2[0][2],skeleton2[0][1], skeleton2[2][2], skeleton2[2][1])

            theta_diff=abs(theta1-theta2)
            print(theta_diff)
            tot_theta_diff+=theta_diff

        if squat>20:
            print("New squat")


    #print("Total mean difference of back angle:",tot_theta_diff/21)
    #print("Total disparity:",tot_disparityP/len(keypositions))
    #print("Total lower disparity:",tot_lower_disparity/len(keypositions))
    #print("Total Mean MPJPE disparity:",tot_disparityM/len(keypositions))
    #print("Total lower Mean MPJPE disparity:",tot_lower_disparityM/len(keypositions))

if __name__ == '__main__':
    main()
