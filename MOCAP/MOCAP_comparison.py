import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json
import MOCAP_alignment as MOCAP
import math

bones={"pelvis": [0,1], "abs": [1,2], "chest": [2,3], "neck": [3,4],
       "Rclavicle":[3,5],"Rshoulder":[5,6],"Rarm":[6,7], "Rforearm":[7,8],
       "Lclavicle":[3,9],"Lshoulder":[9,10], "Larm":[10,11], "Lforearm":[11,12],
       "Rhip":[0,13], "Rthigh":[13,14],"Rshin":[14,15],
       "Lhip":[0,16], "Lthigh":[16,17],"Lshin":[17,18],
       "Rfoot":[15,19],"Lfoot":[18,20]}

lower_bones={"Rhip":[0,1],"Rthigh":[1,3],"Rshin":[3,5], "Rfoot":[5,7],
             "Lhip":[0,2], "Lthigh":[2,4], "Lshin":[4,6], "Lfoot":[6,8]}

lower_body_indices = [0, 13, 16, 14, 17, 15, 18, 19, 20]

def plot_skeletons(skeleton1, skeleton2, skeleton3, skeleton4, pose, title):

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
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.view_init(azim=57, elev=6)

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
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_zlim([-0.3, 0.3])
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.view_init(azim=57, elev=5)
    #plt.savefig(f'MOCAP_reference_sample/MOCAP_reference_sample_{pose}.png')
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
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.view_init(azim=57, elev=6)

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
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.8,0.8])
    ax2.set_ylim([-0.8,0.8])
    ax2.set_zlim([-0.5,1.2])
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.view_init(azim=37, elev=6)

    plt.suptitle(title)
    #plt.savefig(f'MOCAP_lower_reference_sample/MOCAP_lower_reference_sample_{pose}.png')
    plt.show()

def read_skeleton(file_name, frame):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    frame_data = data[int(frame)]
    body_data = frame_data

    skeleton=[]
    for joint in body_data['keypoints']:
        #print(joint['Position'])
        skeleton.append(joint['Position'])

    return np.array(skeleton)

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

    if angle_diff > 90:
        angle_diff=180-angle_diff

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

    keypositions1=MOCAP.main(file_skeleton1)
    keypositions2=MOCAP.main(file_skeleton2)

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

    squat=0

    reference=[]
    sample1=[]
    sample2=[]
    sample3=[]
    sample4=[]
    sample5=[]


    for i,x in enumerate(keypositions):

        if squat<20:
            reference.append(x[0])
            sample3.append(x[1])
        elif squat<40 and squat>=20:
            sample1.append(x[0])
            sample4.append(x[1])
        elif squat>=40 and squat<60:
            sample2.append(x[0])
            sample5.append(x[1])

        squat+=1

    mpjpe_sample1=0
    mpjpe_sample2=0
    mpjpe_sample3=0
    mpjpe_sample4=0
    mpjpe_sample5=0

    lost_frame5=0

    thetaR=0
    theta1=0
    theta2=0
    theta3=0
    theta4=0
    theta5=0
    theta1_diff=0
    theta2_diff=0
    theta3_diff=0
    theta4_diff=0
    theta5_diff=0

    for i in range(len(reference)):
        print(i)
        skeletonR = read_skeleton(file_skeleton1, reference[i])
        skeleton1 = read_skeleton(file_skeleton1, sample1[i])
        skeleton2 = read_skeleton(file_skeleton1, sample2[i])
        skeleton3 = read_skeleton(file_skeleton2, sample3[i])
        skeleton4 = read_skeleton(file_skeleton2, sample4[i])
        skeleton5 = read_skeleton(file_skeleton2, sample5[i])

        if (np.array(skeleton5).shape)==(21,3):
            skeleton5=skeleton5[indices]
            mpjpe_sample1+=MPJPE(skeletonR,skeleton1)
            mpjpe_sample2+=MPJPE(skeletonR,skeleton2)
            mpjpe_sample3+=MPJPE(skeletonR,skeleton3)
            mpjpe_sample4+=MPJPE(skeletonR,skeleton4)
            mpjpe_sample5+=MPJPE(skeletonR,skeleton5)
        else:
            print("perso")
            mpjpe_sample5+=0
            lost_frame5+=1

        thetaR=compute_angle(skeletonR[0][2],skeletonR[0][1], skeletonR[2][2], skeletonR[2][1])
        theta1=compute_angle(skeleton1[0][2],skeleton1[0][1], skeleton1[2][2], skeleton1[2][1])
        theta2=compute_angle(skeleton2[0][2],skeleton2[0][1], skeleton2[2][2], skeleton2[2][1])
        theta3=compute_angle(skeleton3[0][2],skeleton3[0][1], skeleton3[2][2], skeleton3[2][1])
        theta4=compute_angle(skeleton4[0][2],skeleton4[0][1], skeleton4[2][2], skeleton4[2][1])
        theta5=compute_angle(skeleton5[0][2],skeleton5[0][1], skeleton5[2][2], skeleton5[2][1])

        theta1_diff+=abs(thetaR -theta1)
        theta2_diff+=abs(thetaR -theta2)
        theta3_diff+=abs(thetaR -theta3)
        theta4_diff+=abs(thetaR -theta4)
        theta5_diff+=abs(thetaR -theta5)

    print("MPJPE1",str(mpjpe_sample1/len(reference)))
    print("MPJPE2",str(mpjpe_sample2/len(reference)))
    print("MPJPE3",str(mpjpe_sample3/len(reference)))
    print("MPJPE4",str(mpjpe_sample4/len(reference)))
    print("MPJPE5",str(mpjpe_sample5/(len(reference)-lost_frame5)))

    print("THETA1 DIFF",str(theta1_diff/len(reference)))
    print("THETA2 DIFF",str(theta2_diff/len(reference)))
    print("THETA3 DIFF",str(theta3_diff/len(reference)))
    print("THETA4 DIFF",str(theta4_diff/len(reference)))
    print("THETA5 DIFF",str(theta5_diff/len(reference)))

if __name__ == '__main__':
    main()
