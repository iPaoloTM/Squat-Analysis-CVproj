import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import json
import sys
import glob
import sys
import math
from scipy.signal import argrelextrema

bones={"pelvis+abs": [0,1], "chest": [1,2], "neck": [3,26],
       "Rclavicle":[3,11],"Rshoulder":[11,12],"Rarm":[12,13], "Rforearm":[13,14],
       "Lclavicle":[3,4],"Lshoulder":[4,5], "Larm":[5,6], "Lforearm":[6,7],
       "Rhip":[0,22], "Rthigh":[22,23],"Rshin":[23,24],
       "Lhip":[0,18], "Lthigh":[18,19],"Lshin":[19,20],
       "Rfoot":[25,33],"Rankle":[24,33],"Lfoot":[21,32],"Lankle":[20,32]}

def compute_angle(x1,y1,x2,y2,x3,y3,x4,y4):

    if (((x2 - x1)!=0) & ((x4 - x3)!=0)):
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)
    elif (((x2 - x1)==0) & ((x4 - x3)==0)):
        slope1=math.inf
        slope2=math.inf
    elif ((x2-x1)==0):
        slope1=math.inf
        slope2 = (y4 - y3) / (x4 - x3)
    elif ((x4 - x3)==0):
        slope1 = (y2 - y1) / (x2 - x1)
        slope2=math.inf

    angle1 = math.degrees(math.atan(slope1))
    angle2 = math.degrees(math.atan(slope2))

    angle_diff = abs(angle2 - angle1)

    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return 180-angle_diff

def compute_pose(Rarm_angle, Larm_angle, Rleg_angle, Lleg_angle, Rshoulder_angle, Lshoulder_angle, Rshoulder_arm_angle, Lshoulder_arm_angle,skeleton):

    pelvis_y=skeleton[0][1]

    # print("RARM ANGLE:",Rarm_angle)
    # print("RARM ANGLE:",Larm_angle)
    #
    # print("RLEG ANGLE:",Rleg_angle)
    # print("LLEG ANGLE:",Lleg_angle)
    #
    # print("NECK_LSHOULDER ANGLE:",Lshoulder_angle)
    # print("NECK_RSHOULDER ANGLE:",Rshoulder_angle)
    #
    # print("RSHOULDER-ARM ANGLE:",Rshoulder_arm_angle)
    # print("LSHOULDER-ANGLE:",Lshoulder_arm_angle)
    # print("PELVIS POSITION:", )


    if (
        abs(Rarm_angle - 180) <= 10
        and abs(Larm_angle - 180) <= 10
        and abs(Rleg_angle - 180) <= 10
        and abs(Lleg_angle - 180) <= 10
        and abs(Rshoulder_arm_angle - 180) <= 20
        and abs(Lshoulder_arm_angle - 180) <= 20
        and abs((Rshoulder_angle+Lshoulder_angle) - 180) <= 14
    ):
        return "T-POSE"
    elif (
        abs(Rleg_angle - 90) <= 10
        and abs(Lleg_angle - 90) <= 10
        or abs(pelvis_y-skeleton[23][1])<= 0.4
        or abs(pelvis_y-skeleton[19][1])<= 0.4
    ):
        return "INTERMEDIATE"
    else:
        return "-"

def read_skeletons(file_name):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    keypoints = []
    for body in data.values():
        for body_part in body['body_list']:
            keypoints.append(body_part['keypoint'])

    keypoints=np.array(keypoints)

    return keypoints

def plot_pelvis_position(skeletons):
    timestamps = range(len(skeletons))
    pelvis_positions=[]
    Rknee_positions=[]
    Lknee_positions=[]
    knee_positions=[]
    mean=[]

    for skeleton in skeletons:
        if len(skeleton)!=0:
            pelvis_positions.append(skeleton[0][1])
            Rknee_positions.append(skeleton[23][1])
            Lknee_positions.append(skeleton[19][1])

    avg=np.mean(pelvis_positions)

    for i,R in enumerate(Rknee_positions):
        knee_positions.append((Rknee_positions[i]+Lknee_positions[i])/2)

    local_minima_indices = argrelextrema(np.array(pelvis_positions), np.less)[0]

    #post-processing
    local_minima_indices = [i for i in local_minima_indices if pelvis_positions[i] <= avg]

    plt.scatter(local_minima_indices, np.array(pelvis_positions)[local_minima_indices], color='red', label='Local Minima')

    plt.plot(timestamps, pelvis_positions, knee_positions)
    plt.xlabel('Time')
    plt.ylabel('Pelvis Position')
    plt.title('Pelvis and Knee Position over Time')
    plt.gca().legend(('Pelvis(Y)','Knees(Y)'))
    #plt.show()

    return local_minima_indices

def plot_pose(pose_state):

    colors = []
    for state in pose_state:
        if state == "T-POSE":
            colors.append('blue')
        elif state == "INTERMEDIATE":
            colors.append('orange')
        else:
            colors.append('gray')

    temp = [1.0] * len(colors)

    plt.scatter(range(len(colors)), temp, c=colors, label='Pose State')
    plt.xlabel('Time')
    plt.ylabel('Pelvis Position')
    plt.title('Pose State over Time')
    plt.legend()

    #plt.show()

def compute_keypositions(local_minima, pose_state, skeletons):

    local_minima_filtered = []

    #Filtrate positions that are not in the squatting phase
    for idx in local_minima:
        if idx < len(pose_state) and pose_state[idx] == "INTERMEDIATE":
            local_minima_filtered.append(idx)

    phases=[]

    t=0
    phase=-1
    last_seen="-"

    #Trying to map the local minima to the correct squat number
    for i,idx in enumerate(pose_state):
            if pose_state[i]=="INTERMEDIATE":
                if last_seen=="-":
                    phase+=1
                    phases.append([])
                    #new squat zone
                phases[phase].append(i)
            last_seen=pose_state[i]

    keypositions=[]
    #post-processing
    for i,zone in enumerate(phases):
        keypositions.append([])
        for j in local_minima_filtered:
            for k in zone:
                if j==k:
                    keypositions[i].append(j)

    deep_squats_index=[]

    #now we try to recover the position of deep squat
    for i,zone in enumerate(keypositions):
        deep_squats_index.append(0)
        min=math.inf
        min_t=0
        for local_minimum in zone:
            if (skeletons[local_minimum][0][1]<min):
                min=skeletons[local_minimum][0][1]
                deep_squats_index[i]=local_minimum
                #print("the deep squat is at frame "+str(deep_squats_index[i]))

    for i,x in enumerate(deep_squats_index):
        if x==0:
            deep_squats_index.pop(i)

    for i,x in enumerate(deep_squats_index):
        if x==0:
            deep_squats_index.pop(i)

    print(deep_squats_index)

    ################################################################################
    timestamps = range(len(skeletons))
    pelvis_positions=[]
    Rknee_positions=[]
    Lknee_positions=[]
    knee_positions=[]
    for skeleton in skeletons:
        if len(skeleton)!=0:
            pelvis_positions.append(skeleton[0][1])
            Rknee_positions.append(skeleton[14][1])
            Lknee_positions.append(skeleton[16][1])

    for i,R in enumerate(Rknee_positions):
        knee_positions.append((Rknee_positions[i]+Lknee_positions[i])/2)

    plt.scatter(deep_squats_index, np.array(pelvis_positions)[deep_squats_index], color='purple', label='Local Minima')

    #plt.plot(timestamps, pelvis_positions, knee_positions)
    plt.xlabel('Time')
    plt.ylabel('Pelvis Position')
    plt.title('Key positions')
    plt.show()

    return deep_squats_index


def main():

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        print("No file name provided.")
        exit(1)

    skeletons=read_skeletons(file_name)
    #skeleton=skeletons[frame]

    #print(skeleton.shape)

    # # split the points into x, y, z coordinates
    # x = [p[0] for p in skeleton]
    # y = [p[1] for p in skeleton]
    # z = [p[2] for p in skeleton]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(x,y,z)
    #
    # for bone, indices in bones.items():
    #     idx1, idx2 = indices
    #     ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], color='red')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title(f'Frame {frame}')
    #
    # plt.show()

    # skeleton[bones["Rarm"][0]][2] -> THE Z COORD OF THE FIRST POINT OF THE R ARM BONE
    # skeleton[bones["Rarm"][1]][1] -> THE Y COORD OF THE SECOND POINT OF THE R ARM BONE

    pose_state=[]

    for i,skeleton in enumerate(skeletons):
        if (len(skeleton)!=0 and np.array(skeleton).shape==(34,3)):
            Rarm_angle=compute_angle(skeleton[bones["Rarm"][0]][2],skeleton[bones["Rarm"][0]][1],skeleton[bones["Rarm"][1]][2],skeleton[bones["Rarm"][1]][1],skeleton[bones["Rforearm"][0]][2],skeleton[bones["Rforearm"][0]][1],skeleton[bones["Rforearm"][1]][2],skeleton[bones["Rforearm"][1]][1])
            Larm_angle=compute_angle(skeleton[bones["Larm"][0]][2],skeleton[bones["Larm"][0]][1],skeleton[bones["Larm"][1]][2],skeleton[bones["Larm"][1]][1],skeleton[bones["Lforearm"][0]][2],skeleton[bones["Lforearm"][0]][1],skeleton[bones["Lforearm"][1]][2],skeleton[bones["Lforearm"][1]][1])
            Rleg_angle=compute_angle(skeleton[bones["Rthigh"][0]][2],skeleton[bones["Rthigh"][0]][1],skeleton[bones["Rthigh"][1]][2],skeleton[bones["Rthigh"][1]][1],skeleton[bones["Rshin"][0]][2],skeleton[bones["Rshin"][0]][1],skeleton[bones["Rshin"][1]][2],skeleton[bones["Rshin"][1]][1])
            Lleg_angle=compute_angle(skeleton[bones["Lthigh"][0]][2],skeleton[bones["Lthigh"][0]][1],skeleton[bones["Lthigh"][1]][2],skeleton[bones["Lthigh"][1]][1],skeleton[bones["Lshin"][0]][2],skeleton[bones["Lshin"][0]][1],skeleton[bones["Lshin"][1]][2],skeleton[bones["Lshin"][1]][1])
            Rshoulder_angle=compute_angle(skeleton[bones["Rshoulder"][0]][2],skeleton[bones["Rshoulder"][0]][1],skeleton[bones["Rshoulder"][1]][2],skeleton[bones["Rshoulder"][1]][1],skeleton[bones["neck"][0]][2],skeleton[bones["neck"][0]][1],skeleton[bones["neck"][1]][2],skeleton[bones["neck"][1]][1])
            Lshoulder_angle=compute_angle(skeleton[bones["Lshoulder"][0]][2],skeleton[bones["Lshoulder"][0]][1],skeleton[bones["Lshoulder"][1]][2],skeleton[bones["Lshoulder"][1]][1],skeleton[bones["neck"][0]][2],skeleton[bones["neck"][0]][1],skeleton[bones["neck"][1]][2],skeleton[bones["neck"][1]][1])
            Rshoulder_arm_angle=compute_angle(skeleton[bones["Rshoulder"][0]][2],skeleton[bones["Rshoulder"][0]][1],skeleton[bones["Rshoulder"][1]][2],skeleton[bones["Rshoulder"][1]][1],skeleton[bones["Rarm"][0]][2],skeleton[bones["Rarm"][0]][1],skeleton[bones["Rarm"][1]][2],skeleton[bones["Rarm"][1]][1])
            Lshoulder_arm_angle=compute_angle(skeleton[bones["Lshoulder"][0]][2],skeleton[bones["Lshoulder"][0]][1],skeleton[bones["Lshoulder"][1]][2],skeleton[bones["Lshoulder"][1]][1],skeleton[bones["Larm"][0]][2],skeleton[bones["Larm"][0]][1],skeleton[bones["Larm"][1]][2],skeleton[bones["Larm"][1]][1])

            pose_state.append(compute_pose(Rarm_angle,Larm_angle,Rleg_angle,Lleg_angle,Rshoulder_angle,Lshoulder_angle,Rshoulder_arm_angle,Lshoulder_arm_angle,skeleton))

    local_minima=plot_pelvis_position(skeletons)

    plot_pose(pose_state)

    compute_keypositions(local_minima, pose_state, skeletons)


if __name__ == '__main__':
    main()
