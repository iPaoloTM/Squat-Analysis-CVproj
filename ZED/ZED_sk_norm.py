import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

desired_bone_length=1

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
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                skeleton.append(body_part['keypoint'])

    print(skeleton)

    return np.array(skeleton[0])

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

def center_skeleton(skeleton):

    pelvis_position = skeleton[0]

    # Compute the displacement vector
    displacement_vector = -pelvis_position

    for i in range(len(skeleton)):
        skeleton[i] += displacement_vector

    return skeleton

def main():

    skeleton10=np.array([[ 0.37449211, -0.18605915, -3.93861628],
     [ 0.37678632, -0.0322468  ,-3.95772076],
     [ 0.37919202,  0.12155651 ,-3.97688437],
     [ 0.38159779,  0.2753644  ,-3.99604845],
     [ 0.05,  0.2756924  ,-3.99491453],
     [ 0.55293322,  0.27414486 ,-3.99114203],
     [ 0.80835313,  0.26142779 ,-3.98261499],
     [ 1.05404365,  0.24837714 ,-3.98267508],
     [ 1.10318172,  0.24576701 ,-3.982687  ],
     [ 1.20145798,  0.24054676 ,-3.98271108],
     [ 1.14920437,  0.18447715 ,-3.97688961],
     [ 0.8 ,  0.27670813, -3.99739075],
     [ 0.21028852,  0.27825567, -4.00116348],
     [-0.04535139 , 0.28602764, -4.00901127],
     [-0.29106355 , 0.29328951, -4.01935339],
     [-0.340206  ,  0.2947419 , -4.02142191],
     [-0.43849087,  0.29764664, -4.02555895],
     [-0.39127219,  0.23733947, -4.01911116],
     [ 0.46059144, -0.18706027, -3.93633676],
     [ 0.47535822, -0.8, -3.87148285],
     [ 0.48831502 ,-1.5, -3.80730033],
     [ 0.48854268 ,-1.5, -3.67641091],
     [ 0.28839278 ,-0.18505803, -3.9408958 ],
     [ 0.2773416 , -0.8, -3.88261938],
     [ 0.2674329 , -1.00193417, -3.82740903],
     [ 0.26207811, -1.5, -3.69836664],
     [ 0.38707519,  0.40714115, -3.9335146 ],
     [ 0.38865381,  0.45120221, -3.93324351],
     [ 0.41582337,  0.48130298, -3.96423769],
     [ 0.46255547 , 0.45985371, -4.04871941],
     [ 0.36295292,  0.48319328, -3.96359015],
     [ 0.31275585,  0.46520948, -4.04688454],
     [ 0.49255806, -1.10345531, -3.82953596],
     [ 0.26591492, -1.5, -3.85201979]])

    if len(sys.argv) > 4:
        file_skeleton1 = sys.argv[1]
        frame_skeleton1 = sys.argv[2]
        file_skeleton2 = sys.argv[3]
        frame_skeleton2 = sys.argv[4]
    else:
        print("Not enough arguments")
        exit(1)

    skeleton1=[]
    skeleton2=[]

    bones_indexes=[[0,1],[1,2],[2,3],[2,4],[4,5],[5,6],[6,7],[7,8],[8,9],[7,10],[2,11],[11,12],[12,13],[13,14],[14,15],[15,16],[14,17],[3,26],[0,18],[18,19],[19,20],[20,21],[20,32],[0,22],[22,23],[23,24],[24,25],[24,33]]

    skeleton1 = read_skeleton(file_skeleton1, frame_skeleton1)
    skeleton2 = read_skeleton(file_skeleton2, frame_skeleton2)

    bone_length1=0
    bone_length2=0
    bone_length10=0

    for x in bones_indexes:
        bone_length1+=compute_bone_length(skeleton1[x[0]],skeleton1[x[1]])
        bone_length2+=compute_bone_length(skeleton2[x[0]],skeleton2[x[1]])
        bone_length10+=compute_bone_length(skeleton10[x[0]],skeleton10[x[1]])

    print("Skeleton1 bone length:",bone_length1)
    print("Skeleton10 bone length:",bone_length10)

    scaled_skeleton10=scale_skeleton(skeleton10, bone_length10,desired_bone_length)

    scaled_skeleton1=scale_skeleton(skeleton1, bone_length1,desired_bone_length)

    skeleton1=center_skeleton(skeleton1)
    skeleton10=center_skeleton(skeleton10)
    scaled_skeleton1=center_skeleton(scaled_skeleton1)
    scaled_skeleton10=center_skeleton(scaled_skeleton10)

    plot_skeletons2(skeleton1, skeleton10, scaled_skeleton1, scaled_skeleton10, "ecco qui")

    bone_length1=0
    bone_length10=0

    for x in bones_indexes:
        bone_length1+=compute_bone_length(scaled_skeleton1[x[0],scaled_skeleton1[x[1]])
        bone_length10+=compute_bone_length(scaled_skeleton10[x[0]],scaled_skeleton10[x[1]])

    print("Scaled Skeleton1 bone length:",bone_length1)
    print("Scaled Skeleton10 bone length:",bone_length10)


if __name__ == '__main__':
    main()
