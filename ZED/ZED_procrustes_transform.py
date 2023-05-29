import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

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

    x2 = [p[0] for p in skeleton2]
    y2 = [p[1] for p in skeleton2]
    z2 = [p[2] for p in skeleton2]
    ax1.scatter(x2, z2, y2, marker='o', label='Skeleton 2')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_ylim([-6, 4])
    ax1.legend()

    # Plotting the second pair of skeletons
    ax2 = fig.add_subplot(122, projection='3d')
    x3 = [p[0] for p in skeleton3]
    y3 = [p[1] for p in skeleton3]
    z3 = [p[2] for p in skeleton3]
    ax2.scatter(x3, z3, y3, marker='o', label='Aligned Skeleton 1')

    x4 = [p[0] for p in skeleton4]
    y4 = [p[1] for p in skeleton4]
    z4 = [p[2] for p in skeleton4]
    ax2.scatter(x4, z4, y4, marker='o', label='Aligned Skeleton 2')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_ylim([-6, 4])
    ax2.legend()

    plt.suptitle(title)
    plt.show()

def read_skeleton(file_name, frame):
    skeleton=[]
    # Load the second JSON file
    with open('../body_data/first_attempt/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                skeleton.append(body_part['keypoint'])

    return np.array(skeleton[0])

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

    lower_body_indices = [0, 18, 19, 20, 22, 23, 24]

    lower_body_skeleton1=skeleton1[lower_body_indices]
    lower_body_skeleton2=skeleton2[lower_body_indices]

    #plot_skeletons(lower_body_skeleton1,lower_body_skeleton2,"Lower body Skeletons")

    lower_body_skeleton1_2d = lower_body_skeleton1.reshape(7, 3)
    lower_body_skeleton2_2d = lower_body_skeleton2.reshape(7, 3)

    mtx1, mtx2, disparity = procrustes(lower_body_skeleton1_2d, lower_body_skeleton2_2d)

    aligned_lower_body_skeleton1 = mtx1.reshape(7, 3)
    aligned_lower_body_skeleton2 = mtx2.reshape(7, 3)

    plot_skeletons2(lower_body_skeleton1,lower_body_skeleton2,aligned_lower_body_skeleton1,aligned_lower_body_skeleton2,"Aligned Lower body skeletons")

    print("Lower body Disparity:",disparity)

if __name__ == '__main__':
    main()
