import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

def read_skeleton(file_name, frame):
    skeleton=[]
    # Load the second JSON file
    with open('../body_data/first_attempt/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                skeleton.append(body_part['keypoint'])

    return np.array(skeleton)

def compute_bone_length(joint1, joint2):
    """
    Calculate the Euclidean distance between two joints (bone length).

    """
    return np.linalg.norm(joint2 - joint1)

def compute_scaling_factor(total_bone_length, desired_bone_length):
    """
    Compute the scaling factor given the total bone length and the desired bone length.

    """
    return desired_bone_length / total_bone_length

def scale_skeleton(skeleton, scaling_factor):
    """
    Scale a skeleton by applying the scaling factor to each bone length.
    
    """
    scaled_skeleton = skeleton * scaling_factor
    return scaled_skeleton

def main():
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

    for x in bones_indexes:
        bone_length1+=compute_bone_length(skeleton1[0][x[0]],skeleton1[0][x[1]])
        bone_length2+=compute_bone_length(skeleton2[0][x[0]],skeleton2[0][x[1]])

    print("Skeleton1 bone length:",bone_length1)
    print("Skeleton2 bone length:",bone_length2)


if __name__ == '__main__':
    main()
