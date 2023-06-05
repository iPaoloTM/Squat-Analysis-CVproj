import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

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

def plot_skeletons(skeleton1, skeleton2, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(skeleton1[:, 0], skeleton1[:, 1], skeleton1[:, 2], c='blue', label='Skeleton 1')
    ax.scatter(skeleton2[:, 0], skeleton2[:, 1], skeleton2[:, 2], c='red', label='Skeleton 2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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
    ax1.set_xlim([-0.6, 0.6])
    ax1.set_ylim([-1.5, 0.8])
    ax1.set_zlim([-0.5, 0.5])
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
    ax2.set_xlim([-0.6, 0.6])
    ax2.set_ylim([-1.5, 0.8])
    ax2.set_zlim([-0.5, 0.5])
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
    ax1.set_xlim([-0.6, 0.6])
    ax1.set_ylim([-1.5, 0.8])
    ax1.set_zlim([-0.5, 0.5])
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
    ax2.set_xlim([-0.6, 0.6])
    ax2.set_ylim([-1.5, 0.8])
    ax2.set_zlim([-0.5, 0.5])
    ax2.legend()

    plt.suptitle(title)
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

    print("Skeleton1",skeleton1)
    print("Skeleton2",skeleton2)

    #plot_skeletons(skeleton1,skeleton2,"Original Skeletons")

    # Reshape the arrays for Procrustes transformation
    skeleton1_2d = skeleton1.reshape(34, 3)
    skeleton2_2d = skeleton2.reshape(34, 3)

    mtx1, mtx2, disparity = procrustes(skeleton1_2d, skeleton2_2d)
    aligned_skeleton1 = mtx1.reshape(34, 3)
    aligned_skeleton2 = mtx2.reshape(34, 3)

    plot_skeletons2(skeleton1,skeleton2,aligned_skeleton1,aligned_skeleton2,"Aligned Skeletons")

    print("General Disparity:",disparity)

    lower_body_skeleton1=skeleton1[lower_body_indices]
    lower_body_skeleton2=skeleton2[lower_body_indices]

    lower_body_skeleton1 = center_skeleton(lower_body_skeleton1)
    lower_body_skeleton2 = center_skeleton(lower_body_skeleton2)

    #plot_skeletons(lower_body_skeleton1,lower_body_skeleton2,"Lower body Skeletons")

    lower_body_skeleton1_2d = lower_body_skeleton1.reshape(11, 3)
    lower_body_skeleton2_2d = lower_body_skeleton2.reshape(11, 3)

    mtx1, mtx2, disparity = procrustes(lower_body_skeleton1_2d, lower_body_skeleton2_2d)

    aligned_lower_body_skeleton1 = mtx1.reshape(11, 3)
    aligned_lower_body_skeleton2 = mtx2.reshape(11, 3)

    plot_skeletons3(lower_body_skeleton1,lower_body_skeleton2,aligned_lower_body_skeleton1,aligned_lower_body_skeleton2,"Aligned Lower body skeletons")

    print("Lower body Disparity:",disparity)

if __name__ == '__main__':
    main()
