import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

bones={"pelvis": [0,1], "abs": [1,2], "chest": [2,3], "neck": [3,4],
       "Rclavicle":[3,5],"Rshoulder":[5,6],"Rarm":[6,7], "Rforearm":[7,8],
       "Lclavicle":[3,9],"Lshoulder":[9,10], "Larm":[10,11], "Lforearm":[11,12],
       "Rhip":[0,13], "Rthigh":[13,14],"Rshin":[14,15],
       "Lhip":[0,16], "Lthigh":[16,17],"Lshin":[17,18],
       "Rfoot":[15,19],"Lfoot":[18,20]}

lower_bones={"Rhip":[0,1],"Rthigh":[1,3],"Rshin":[3,5], "Rfoot":[5,7],
             "Lhip":[0,2], "Lthigh":[2,4], "Lshin":[4,6], "Lfoot":[6,8]}

lower_body_indices = [0, 13, 16, 14, 17, 15, 18, 19, 20]

def plot_skeletons(skeleton1, skeleton2, title):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1= [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]

    ax.scatter(x1, z1,  y1, marker='o', label='Skeleton 1')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]

    ax.scatter(x2, z2, y2, marker='o', label='Skeleton 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0, 1.8])
    ax.legend()
    plt.title(title)
    plt.show()

def plot_skeletons2(skeleton1, skeleton2, skeleton3, skeleton4, title):

    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(x1, z1, y1, marker='o', label='Skeleton 1')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax1.plot([x1[idx1], x1[idx2]], [z1[idx1], z1[idx2]], [y1[idx1], y1[idx2]], color='blue')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', label='Skeleton 2')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='orange')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-0.8, 0.8])
    ax1.set_ylim([-0.8, 0.8])
    ax1.set_zlim([-1.8, 0.8])
    ax1.legend()

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', label='Aligned Skeleton 1')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax2.plot([x3[idx1], x3[idx2]], [z3[idx1], z3[idx2]], [y3[idx1], y3[idx2]], color='blue')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', label='Aligned Skeleton 2')

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='orange')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.4, 0.4])
    ax2.set_ylim([-0.4, 0.4])
    ax2.set_zlim([-0.7, 0.5])
    ax2.legend()

    plt.suptitle(title)
    plt.show()

def plot_skeletons3(skeleton1, skeleton2, skeleton3, skeleton4, title):


    fig = plt.figure(figsize=(12, 6))

    # Plotting the first pair of skeletons
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = [p[0] for p in skeleton1]
    y1 = [p[1] for p in skeleton1]
    z1 = [p[2] for p in skeleton1]
    ax1.scatter(x1, z1, y1, marker='o', label='Skeleton 1')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax1.plot([x1[idx1], x1[idx2]], [z1[idx1], z1[idx2]], [y1[idx1], y1[idx2]], color='blue')

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', label='Skeleton 2')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax1.plot([x2[idx1], x2[idx2]], [z2[idx1], z2[idx2]], [y2[idx1], y2[idx2]], color='orange')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-0.8, 0.8])
    ax1.set_ylim([-0.8, 0.8])
    ax1.set_zlim([-1.8, 0.8])
    ax1.legend()

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', label='Aligned Skeleton 1')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax2.plot([x3[idx1], x3[idx2]], [z3[idx1], z3[idx2]], [y3[idx1], y3[idx2]], color='blue')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', label='Aligned Skeleton 2')

    for bone, indices in lower_bones.items():
        idx1, idx2 = indices
        ax2.plot([x4[idx1], x4[idx2]], [z4[idx1], z4[idx2]], [y4[idx1], y4[idx2]], color='orange')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.4, 0.4])
    ax2.set_ylim([-0.4, 0.4])
    ax2.set_zlim([-0.7, 0.5])
    ax2.legend()

    plt.suptitle(title)
    plt.show()

def read_skeleton(file_name, frame):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    frame_data = data[int(frame)]
    body_data = frame_data

    skeleton=[]
    for joint in body_data['keypoints']:
        print(joint['Position'])
        skeleton.append(joint['Position'])

    return np.array(skeleton)

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

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

    skeleton1 = read_skeleton(file_skeleton1, frame_skeleton1)
    skeleton2 = read_skeleton(file_skeleton2, frame_skeleton2)

    skeleton1 = center_skeleton(skeleton1)
    skeleton2 = center_skeleton(skeleton2)

    # Padding the smaller skeleton with zeros to match the shape of the larger skeleton
    max_points = max(skeleton1.shape[0], skeleton2.shape[0])
    if skeleton1.shape[0] < max_points:
        skeleton1 = np.pad(skeleton1, ((0, max_points - skeleton1.shape[0]), (0, 0)), mode='constant')
    elif skeleton2.shape[0] < max_points:
        skeleton2 = np.pad(skeleton2, ((0, max_points - skeleton2.shape[0]), (0, 0)), mode='constant')

    #print("Skeleton1.shape",skeleton1)
    #print("Skeleton2.shape",skeleton2)

    #plot_skeletons(skeleton1,skeleton2,"Original Skeletons")

    # Reshape the arrays for Procrustes transformation
    skeleton1_2d = skeleton1.reshape(21, 3)
    skeleton2_2d = skeleton2.reshape(21, 3)

    mtx1, mtx2, disparity = procrustes(skeleton1_2d, skeleton2_2d)
    aligned_skeleton1 = mtx1.reshape(21, 3)
    aligned_skeleton2 = mtx2.reshape(21, 3)

    plot_skeletons2(skeleton1,skeleton2,aligned_skeleton1,aligned_skeleton2,"Aligned Skeletons")

    print("General Disparity:",disparity)

    lower_body_skeleton1=skeleton1[lower_body_indices]
    lower_body_skeleton2=skeleton2[lower_body_indices]

    lower_body_skeleton1 = center_skeleton(lower_body_skeleton1)
    lower_body_skeleton2 = center_skeleton(lower_body_skeleton2)

    lower_body_skeleton1_2d = lower_body_skeleton1.reshape(9, 3)
    lower_body_skeleton2_2d = lower_body_skeleton2.reshape(9, 3)

    mtx1, mtx2, disparity = procrustes(lower_body_skeleton1_2d, lower_body_skeleton2_2d)

    aligned_lower_body_skeleton1 = mtx1.reshape(9, 3)
    aligned_lower_body_skeleton2 = mtx2.reshape(9, 3)

    plot_skeletons3(lower_body_skeleton1,lower_body_skeleton2,aligned_lower_body_skeleton1,aligned_lower_body_skeleton2,"Aligned Lower body skeletons")

    print("Lower body Disparity:",disparity)

if __name__ == '__main__':
    main()
