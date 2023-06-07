import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

ZED_bones={"pelvis+abs": [0,1], "chest": [1,3], "neck": [3,26],
       "Rclavicle":[3,11],"Rshoulder":[11,12],"Rarm":[12,13], "Rforearm":[13,14],
       "Lclavicle":[3,4],"Lshoulder":[4,5], "Larm":[5,6], "Lforearm":[6,7],
       "Rhip":[0,22], "Rthigh":[22,23],"Rshin":[23,24],
       "Lhip":[0,18], "Lthigh":[18,19],"Lshin":[19,20],
       "Rfoot":[24,25],"Lfoot":[21,26]}

ZED_index=[0,1,3,4,5,6,7,11,12,13,14,22,23,24,18,19,20,25,21,26]

MOCAP_bones={"pelvis": [0,2], "chest": [2,3], "neck": [3,4],
       "Rclavicle":[3,5],"Rshoulder":[5,6],"Rarm":[6,7], "Rforearm":[7,8],
       "Lclavicle":[3,9],"Lshoulder":[9,10], "Larm":[10,11], "Lforearm":[11,12],
       "Rhip":[0,13], "Rthigh":[13,14],"Rshin":[14,15],
       "Lhip":[0,16], "Lthigh":[16,17],"Lshin":[17,18],
       "Rfoot":[15,19],"Lfoot":[18,20]}

# ZED_index=[0,1,3,4,5,6,7,11,12,13,14,22,23,24,18,19,20,25,21,26]
#
# MOCAP_mapping=[0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,4]



MOCAP_mapping=[0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,4]


def plot_skeletons(skeleton1, skeleton2, skeleton3, skeleton4, title):

    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(z1, x1, y1, marker='o', color='green', label='ZED')

    # ax1.plot([skeleton1[0][2], skeleton1[1][2]], [skeleton1[0][0], skeleton1[1][0]], [skeleton1[0][1], skeleton1[1][1]], color='green')
    # ax1.plot([skeleton1[1][2], skeleton1[2][2]], [skeleton1[1][0], skeleton1[2][0]], [skeleton1[1][1], skeleton1[2][1]], color='green')
    # ax1.plot([skeleton1[2][2], skeleton1[3][2]], [skeleton1[2][0], skeleton1[3][0]], [skeleton1[2][1], skeleton1[3][1]], color='green')
    # ax1.plot([skeleton1[3][2], skeleton1[4][2]], [skeleton1[3][0], skeleton1[4][0]], [skeleton1[3][1], skeleton1[4][1]], color='green')
    # ax1.plot([skeleton1[4][2], skeleton1[5][2]], [skeleton1[4][0], skeleton1[5][0]], [skeleton1[4][1], skeleton1[5][1]], color='green')
    # ax1.plot([skeleton1[5][2], skeleton1[6][2]], [skeleton1[5][0], skeleton1[6][0]], [skeleton1[5][1], skeleton1[6][1]], color='green')
    # ax1.plot([skeleton1[3][2], skeleton1[7][2]], [skeleton1[3][0], skeleton1[7][0]], [skeleton1[3][1], skeleton1[7][1]], color='green')
    # ax1.plot([skeleton1[7][2], skeleton1[8][2]], [skeleton1[7][0], skeleton1[8][0]], [skeleton1[7][1], skeleton1[8][1]], color='green')
    # ax1.plot([skeleton1[8][2], skeleton1[9][2]], [skeleton1[8][0], skeleton1[9][0]], [skeleton1[8][1], skeleton1[9][1]], color='green')
    # ax1.plot([skeleton1[9][2], skeleton1[10][2]], [skeleton1[9][0], skeleton1[10][0]], [skeleton1[9][1], skeleton1[10][1]], color='green')
    # ax1.plot([skeleton1[0][2], skeleton1[11][2]], [skeleton1[0][0], skeleton1[11][0]], [skeleton1[0][1], skeleton1[11][1]], color='green')
    # ax1.plot([skeleton1[11][2], skeleton1[12][2]], [skeleton1[11][0], skeleton1[12][0]], [skeleton1[11][1], skeleton1[12][1]], color='green')
    # ax1.plot([skeleton1[12][2], skeleton1[13][2]], [skeleton1[12][0], skeleton1[13][0]], [skeleton1[12][1], skeleton1[13][1]], color='green')
    # ax1.plot([skeleton1[0][2], skeleton1[14][2]], [skeleton1[0][0], skeleton1[14][0]], [skeleton1[0][1], skeleton1[14][1]], color='green')
    # ax1.plot([skeleton1[14][2], skeleton1[15][2]], [skeleton1[14][0], skeleton1[15][0]], [skeleton1[14][1], skeleton1[15][1]], color='green')
    # ax1.plot([skeleton1[15][2], skeleton1[16][2]], [skeleton1[15][0], skeleton1[16][0]], [skeleton1[15][1], skeleton1[16][1]], color='green')
    # ax1.plot([skeleton1[13][2], skeleton1[17][2]], [skeleton1[13][0], skeleton1[17][0]], [skeleton1[13][1], skeleton1[17][1]], color='green')
    # ax1.plot([skeleton1[16][2], skeleton1[18][2]], [skeleton1[16][0], skeleton1[18][0]], [skeleton1[16][1], skeleton1[18][1]], color='green')
    # ax1.plot([skeleton1[2][2], skeleton1[19][2]], [skeleton1[2][0], skeleton1[19][0]], [skeleton1[2][1], skeleton1[19][1]], color='green')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', color='orange', label='MOCAP')

    # for bone, indices in MOCAP_bones.items():
    #     idx1, idx2 = indices
    #     ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='orange')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.legend()

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', color='green', label='ZED')

    # ax2.plot([skeleton3[0][2], skeleton3[1][2]], [skeleton3[0][0], skeleton3[1][0]], [skeleton3[0][1], skeleton3[1][1]], color='green')
    # ax2.plot([skeleton3[1][0], skeleton3[2][0]], [skeleton3[1][1], skeleton3[2][1]], [skeleton3[1][2], skeleton3[2][2]], color='green')
    # ax2.plot([skeleton3[2][0], skeleton3[3][0]], [skeleton3[2][1], skeleton3[3][1]], [skeleton3[2][2], skeleton3[3][2]], color='green')
    # ax2.plot([skeleton3[3][0], skeleton3[4][0]], [skeleton3[3][1], skeleton3[4][1]], [skeleton3[3][2], skeleton3[4][2]], color='green')
    # ax2.plot([skeleton3[4][0], skeleton3[5][0]], [skeleton3[4][1], skeleton3[5][1]], [skeleton3[4][2], skeleton3[5][2]], color='green')
    # ax2.plot([skeleton3[5][0], skeleton3[6][0]], [skeleton3[5][1], skeleton3[6][1]], [skeleton3[5][2], skeleton3[6][2]], color='green')
    # ax2.plot([skeleton3[3][0], skeleton3[7][0]], [skeleton3[3][1], skeleton3[7][1]], [skeleton3[3][2], skeleton3[7][2]], color='green')
    # ax2.plot([skeleton3[7][0], skeleton3[8][0]], [skeleton3[7][1], skeleton3[8][1]], [skeleton3[7][2], skeleton3[8][2]], color='green')
    # ax2.plot([skeleton3[8][0], skeleton3[9][0]], [skeleton3[8][1], skeleton3[9][1]], [skeleton3[8][2], skeleton3[9][2]], color='green')
    # ax2.plot([skeleton3[9][0], skeleton3[10][0]], [skeleton3[9][1], skeleton3[10][1]], [skeleton3[9][2], skeleton3[10][2]], color='green')
    # ax2.plot([skeleton3[0][0], skeleton3[11][0]], [skeleton3[0][1], skeleton3[11][1]], [skeleton3[0][2], skeleton3[11][2]], color='green')
    # ax2.plot([skeleton3[11][0], skeleton3[12][0]], [skeleton3[11][1], skeleton3[12][1]], [skeleton3[11][2], skeleton3[12][2]], color='green')
    # ax2.plot([skeleton3[12][0], skeleton3[13][0]], [skeleton3[12][1], skeleton3[13][1]], [skeleton3[12][2], skeleton3[13][2]], color='green')
    # ax2.plot([skeleton3[0][0], skeleton3[14][0]], [skeleton3[0][1], skeleton3[14][1]], [skeleton3[0][2], skeleton3[14][2]], color='green')
    # ax2.plot([skeleton3[14][0], skeleton3[15][0]], [skeleton3[14][1], skeleton3[15][1]], [skeleton3[14][2], skeleton3[15][2]], color='green')
    # ax2.plot([skeleton3[15][0], skeleton3[16][0]], [skeleton3[15][1], skeleton3[16][1]], [skeleton3[15][2], skeleton3[16][2]], color='green')
    # ax2.plot([skeleton3[13][0], skeleton3[17][0]], [skeleton3[13][1], skeleton3[17][1]], [skeleton3[13][2], skeleton3[17][2]], color='green')
    # ax2.plot([skeleton3[16][0], skeleton3[18][0]], [skeleton3[16][1], skeleton3[18][1]], [skeleton3[16][2], skeleton3[18][2]], color='green')
    # ax2.plot([skeleton3[2][0], skeleton3[19][0]], [skeleton3[2][1], skeleton3[19][1]], [skeleton3[2][2], skeleton3[19][2]], color='green')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', color='orange', label='MOCAP')

    # for bone, indices in MOCAP_bones.items():
    #     idx1, idx2 = indices
    #     ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='orange')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_zlim([-0.3, 0.3])
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

def main():
    if len(sys.argv) > 4:
        file_skeleton1 = sys.argv[1]
        frame_skeleton1 = sys.argv[2]
        file_skeleton2 = sys.argv[3]
        frame_skeleton2 = sys.argv[4]
        print("Computing alignment between "+file_skeleton1+" at frame "+frame_skeleton1+" and "+file_skeleton2+" at frame "+frame_skeleton2)
    else:
        print("Not enough arguments")
        exit(1)

    skeleton1 = read_skeletonZED(file_skeleton1, frame_skeleton1)
    skeleton1 = center_skeleton(skeleton1)

    skeleton_new_ZED=skeleton1[ZED_index]

    skeleton2 = read_skeletonMOCAP(file_skeleton2, frame_skeleton2)

    skeleton2 = center_skeleton(skeleton2)

    skeleton_new_MOCAP=skeleton2[MOCAP_mapping]


    # Reshape the arrays for Procrustes transformation
    skeleton1_2d = skeleton_new_ZED.reshape(20, 3)
    skeleton2_2d = skeleton_new_MOCAP.reshape(20, 3)

    mtx1, mtx2, disparity = procrustes(skeleton1_2d, skeleton2_2d)
    aligned_skeleton1 = mtx1.reshape(20, 3)
    aligned_skeleton2 = mtx2.reshape(20, 3)


    plot_skeletons(skeleton_new_ZED,skeleton_new_MOCAP,aligned_skeleton1,aligned_skeleton2,"Aligned Skeletons")

    print("General Disparity:",disparity)


if __name__ == '__main__':
    main()
