import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import json
import sys
import glob
import sys
import math
from scipy.signal import argrelextrema

'''
ZED Skeleton with 34 joints
'''

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

    '''
    Critera:
    T_POSE: (starting position) the skeleton has the arms, forearms and shoulders on the same horizontal line. Same with legs (thighs + shins)
            and the neck must form a 90 degrees angle with both the shoulders
    INTERMEDIATE: (when the squat is about to begin/end) the thighs and the shins form a 90 degrees angle and the pelvis y-coord
                  is 'near' the y-coord of both the knees
    -: everything else
    '''

    pelvis_y=skeleton[0][1]

    # print("RARM ANGLE:",Rarm_angle)
    # print("LARM ANGLE:",Larm_angle)
    #
    # print("RLEG ANGLE:",Rleg_angle)
    # print("LLEG ANGLE:",Lleg_angle)
    #
    # print("NECK_LSHOULDER ANGLE:",Lshoulder_angle)
    # print("NECK_RSHOULDER ANGLE:",Rshoulder_angle)
    #
    # print("RSHOULDER-ARM ANGLE:",Rshoulder_arm_angle)
    # print("LSHOULDER-ANGLE:",Lshoulder_arm_angle)
    # print("PELVIS POSITION:", pelvis_y )

    if (
        abs(Rarm_angle - 180) <= 10
        and abs(Larm_angle - 180) <= 10
        and abs(Rleg_angle - 180) <= 10
        and abs(Lleg_angle - 180) <= 10
        and abs(Rshoulder_arm_angle - 180) <= 50
        and abs(Lshoulder_arm_angle - 180) <= 50
        and abs((Rshoulder_angle+Lshoulder_angle) - 180) <= 14
    ):
        return "T-POSE"
    elif (
        abs(Rleg_angle - 90) <= 10
        and abs(Lleg_angle - 90) <= 10
        or abs(pelvis_y-skeleton[23][1])<= 0.35
        or abs(pelvis_y-skeleton[19][1])<= 0.35
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

def compute_local_minima(skeletons):

    '''
    Assumption: we expect a function with many local minima, corresponding to the different deep squat phases
    So we search for the local minima and then isolate a unique minimum for each squat
    In this function we plot the positions timeline, the variation of the y coordinate of the pelvis joint
    and of the knees (average of the two), showing the local minima computed by np.lextrema and finding the real minimum.
    '''
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

    #plt.scatter(local_minima_indices, np.array(pelvis_positions)[local_minima_indices], color='red', label='Local Minima')

    plt.plot(timestamps, pelvis_positions)
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

    #plt.show()

def compute_squat_positions(local_minima, pose_state, skeletons):

    '''
    This function return the time instans of the relevant position for the squat sequence
    T-POSE: the beginning of the sequence
    INTERMEDIATE: the squat is about to start or just finished
    INTERMEDIATE_UP: in the middle of the ascending phase for the squat
    INTERMEDIATE_DOWN: in the middle of the descending phase for the squat
    '''

    pose_index=[]
    Tpose_index=[]
    deep_squats_index=[]
    local_minima_filtered = []

    '''
    Here we filter the local minima that are not in the nearby of the squat action
    '''
    for idx in local_minima:
        if idx < len(pose_state) and pose_state[idx] == "INTERMEDIATE":
            local_minima_filtered.append(idx)

    phases=[]
    phase=-1
    last_seen="-"

    '''
    Here we map each local minima to its respective squat action
    and we also begin to populate the final array: pose_index with the
    start and end of a squat action phase
    '''
    for i,idx in enumerate(pose_state):
        pose_index.append([])
        if pose_state[i]=="T-POSE":
            Tpose_index.append(i)
        elif pose_state[i]=="INTERMEDIATE":
            if last_seen=="-":
                pose_index[i]="intermediate"
                phase+=1
                phases.append([])
                #new squat zone
            phases[phase].append(i)
        elif pose_state[i]=="-":
            if last_seen=="INTERMEDIATE":
                pose_index[i-1]="intermediate"
        last_seen=pose_state[i]

    '''
    Find and mark the T pose instant with an average of the instants when the person is int T-pose

    '''
    t_pose_index = int(np.mean(Tpose_index))

    pose_index[t_pose_index]="T-POSE"

    '''
      SQUAT POSITIONS
      The idea here is to iterate on the time instants and search only for the real minimum in the squatting phase,
      being aware of the number of the squatting action
    '''
    temp_keypositions=[]
    #post-processing
    for i,zone in enumerate(phases):
        temp_keypositions.append([])
        for j in local_minima_filtered:
            for k in zone:
                if j==k:
                    temp_keypositions[i].append(j)

    #now we try to recover the position of deep squat
    for i,zone in enumerate(temp_keypositions):
        deep_squats_index.append(0)
        min=math.inf
        min_t=0
        for local_minimum in zone:
            if (skeletons[local_minimum][0][1]<min):
                min=skeletons[local_minimum][0][1]
                deep_squats_index[i]=local_minimum
                #print("the deep squat is at frame "+str(deep_squats_index[i]))

    '''
    Remove 0 from deep_squats_index
    '''

    for i,x in enumerate(deep_squats_index):
        if x==0:
            deep_squats_index.pop(i)

    for i,x in enumerate(deep_squats_index):
        if x==0:
            deep_squats_index.pop(i)

    for x in deep_squats_index:
        pose_index[x]="Squat"

    #print(deep_squats_index)

    '''
       T-POSITIONS
        We retrieve two intermediate positions between the T-pose and the start of the squat sequences
    '''

    i=0
    k=-1
    j=0
    for i,x in enumerate(pose_index):
        if x=='T-POSE':

            k=i
        elif x=='intermediate' and k!=-1:
            pose_index[k+(int((i-k)/3))]='Tintermediate_1'
            pose_index[k+int(2*(i-k)/3)]='Tintermediate_2'
            k=-1
            break

    '''
       INTERMEDIATE POSITIONS
       Now we understand when the squatting action is about to start or when it's finished, and mark the corresponding
       time instant. We also mark two medium positions for both descending (intermediate_down) and ascending phase (intermediate_up)
    '''

    i=0
    j=0
    begin=0 #the y position of the pelvis when beginning squatting
    confidence_down=0.0058

    while j<len(deep_squats_index) and i<len(pose_index):
        if pose_index[i]=='intermediate' and i<deep_squats_index[j]:
            begin=i
        if pose_index[i]=='Squat':
            dist=abs(skeletons[begin][0][1]-skeletons[i][0][1])
            pos1=skeletons[begin][0][1]-((dist)/3)
            pos2=skeletons[begin][0][1]-(((dist)*2)/3)
            x=i
            while x>begin:
                if abs(skeletons[x][0][1]-pos1)<confidence_down:
                    #print("DOWN 1 "+str(j))
                    pose_index[x]='intermediate_down1'
                    pos1=100 #stop searchiing for another match
                elif abs(skeletons[x][0][1]-pos2)<confidence_down:
                    #print("DOWN 2 "+str(j))
                    pose_index[x]='intermediate_down2'
                    pos2=100

                x-=1
            j+=1 #next squat
        i+=1 #next frame


    i=len(pose_index)-1
    j=len(deep_squats_index)-1

    finish=0 #the y position of the pelvis when finishing squatting
    confidence_up=0.0071

    while j>-1 and i>0:
        if pose_index[i]=='intermediate' and i>deep_squats_index[j]:
            finish=i
        if pose_index[i]=='Squat':
            # print("------------------------")
            dist=abs(skeletons[finish][0][1]-skeletons[i][0][1])
            pos1=skeletons[i][0][1]+((dist)/3)
            pos2=skeletons[i][0][1]+(((dist)*2)/3)
            # print("DIST=",str(dist))
            # print("pos1=",str(pos1))
            # print("pos2=",str(pos2))
            # print("------------------------")
            x=i
            while x<finish:
                #print(f"({skeletons[x][0][1]})-({pos1})=",str(abs(skeletons[x][0][1]-pos1)))
                #print(f"({skeletons[x][0][1]})-({pos2})=",str(abs(skeletons[x][0][1]-pos1)))
                if abs(skeletons[x][0][1]-pos1)<confidence_up:
                    #print("UP 1 "+str(j))
                    pose_index[x]='intermediate_up1'
                    pos1=100 #stop searchiing for another match
                elif abs(skeletons[x][0][1]-pos2)<confidence_up:
                    #print("UP 2 "+str(j))
                    pose_index[x]='intermediate_up2'
                    pos2=100

                x+=1
            #xxexit(0)
            j-=1
        i-=1

    pose_index2=[]
    for i,x in enumerate(pose_index):
        if x!=[]:
            pose_index2.append([i,x])

    print(np.array(pose_index2).shape)

    ################################################################################
    pelvis_positions=[]

    for skeleton in skeletons:
        if len(skeleton)!=0:
            pelvis_positions.append(skeleton[0][1])

    indici=[point[0] for point in pose_index2]

    plt.scatter(indici, np.array(pelvis_positions)[indici], color='green', label='Key Positions')

    #plt.plot(timestamps, pelvis_positions, knee_positions)
    plt.xlabel('Time')
    plt.ylabel('Pelvis Position')
    plt.title('Key positions')
    plt.show()

    return indici


def main():

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        #frame = int(sys.argv[2])
    else:
        print("No file name provided.")
        exit(1)

    skeletons=read_skeletons(file_name)
    # skeleton=skeletons[frame]
    #
    # print(skeleton.shape)
    #
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
    # plt.show()


    '''
    THESE ARE ANGLES IN THE PLANE XY
    skeleton[bones["Rarm"][0]][0] -> THE X COORD OF THE FIRST POINT OF THE R ARM BONE
    skeleton[bones["Rarm"][1]][1] -> THE Y COORD OF THE SECOND POINT OF THE R ARM BONE
    '''
    pose_state=[]

    for i,skeleton in enumerate(skeletons):
        if (len(skeleton)!=0 and np.array(skeleton).shape==(34,3)):
            Rarm_angle=compute_angle(skeleton[bones["Rarm"][0]][0],skeleton[bones["Rarm"][0]][1],skeleton[bones["Rarm"][1]][0],skeleton[bones["Rarm"][1]][1],skeleton[bones["Rforearm"][0]][0],skeleton[bones["Rforearm"][0]][1],skeleton[bones["Rforearm"][1]][0],skeleton[bones["Rforearm"][1]][1])
            Larm_angle=compute_angle(skeleton[bones["Larm"][0]][0],skeleton[bones["Larm"][0]][1],skeleton[bones["Larm"][1]][0],skeleton[bones["Larm"][1]][1],skeleton[bones["Lforearm"][0]][0],skeleton[bones["Lforearm"][0]][1],skeleton[bones["Lforearm"][1]][0],skeleton[bones["Lforearm"][1]][1])
            Rleg_angle=compute_angle(skeleton[bones["Rthigh"][0]][0],skeleton[bones["Rthigh"][0]][1],skeleton[bones["Rthigh"][1]][0],skeleton[bones["Rthigh"][1]][1],skeleton[bones["Rshin"][0]][0],skeleton[bones["Rshin"][0]][1],skeleton[bones["Rshin"][1]][0],skeleton[bones["Rshin"][1]][1])
            Lleg_angle=compute_angle(skeleton[bones["Lthigh"][0]][0],skeleton[bones["Lthigh"][0]][1],skeleton[bones["Lthigh"][1]][0],skeleton[bones["Lthigh"][1]][1],skeleton[bones["Lshin"][0]][0],skeleton[bones["Lshin"][0]][1],skeleton[bones["Lshin"][1]][0],skeleton[bones["Lshin"][1]][1])
            Rshoulder_angle=compute_angle(skeleton[bones["Rclavicle"][0]][0],skeleton[bones["Rclavicle"][0]][1],skeleton[bones["Rclavicle"][1]][0],skeleton[bones["Rclavicle"][1]][1],skeleton[bones["neck"][0]][0],skeleton[bones["neck"][0]][1],skeleton[bones["neck"][1]][0],skeleton[bones["neck"][1]][1])
            Lshoulder_angle=compute_angle(skeleton[bones["Lclavicle"][0]][0],skeleton[bones["Lclavicle"][0]][1],skeleton[bones["Lclavicle"][1]][0],skeleton[bones["Lclavicle"][1]][1],skeleton[bones["neck"][0]][0],skeleton[bones["neck"][0]][1],skeleton[bones["neck"][1]][0],skeleton[bones["neck"][1]][1])
            Rshoulder_arm_angle=compute_angle(skeleton[bones["Rshoulder"][0]][0],skeleton[bones["Rshoulder"][0]][1],skeleton[bones["Rshoulder"][1]][0],skeleton[bones["Rshoulder"][1]][1],skeleton[bones["Rarm"][0]][0],skeleton[bones["Rarm"][0]][1],skeleton[bones["Rarm"][1]][0],skeleton[bones["Rarm"][1]][1])
            Lshoulder_arm_angle=compute_angle(skeleton[bones["Lshoulder"][0]][0],skeleton[bones["Lshoulder"][0]][1],skeleton[bones["Lshoulder"][1]][0],skeleton[bones["Lshoulder"][1]][1],skeleton[bones["Larm"][0]][0],skeleton[bones["Larm"][0]][1],skeleton[bones["Larm"][1]][0],skeleton[bones["Larm"][1]][1])

            pose_state.append(compute_pose(Rarm_angle,Larm_angle,Rleg_angle,Lleg_angle,Rshoulder_angle,Lshoulder_angle,Rshoulder_arm_angle,Lshoulder_arm_angle,skeleton))

    local_minima=compute_local_minima(skeletons)

    plot_pose(pose_state)

    print(compute_squat_positions(local_minima, pose_state, skeletons))


if __name__ == '__main__':
    main()
