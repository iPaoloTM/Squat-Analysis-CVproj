import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys
import math

bones={"pelvis": [0,1], "abs": [1,2], "chest": [2,3], "neck": [3,4],
       "Rclavicle":[3,5],"Rshoulder":[5,6],"Rarm":[6,7], "Rforearm":[7,8],
       "Lclavicle":[3,9],"Lshoulder":[9,10], "Larm":[10,11], "Lforearm":[11,12],
       "Rhip":[0,13], "Rthigh":[13,14],"Rshin":[14,15],
       "Lhip":[0,16], "Lthigh":[16,17],"Lshin":[17,18],
       "Rfoot":[15,19],"Lfoot":[18,20]}

def compute_angle(x1,y1,x2,y2,x3,y3,x4,y4):

    if (((x2 - x1)!=0) & ((x4 - x3)!=0)):
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)
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

def compute_T_pose(Rarm_angle, Larm_angle, Rleg_angle, Lleg_angle, Rshoulder_angle, Lshoulder_angle):

    # print("RARM ANGLE:",Rarm_angle)
    # print("RARM ANGLE:",Larm_angle)
    #
    # print("RLEG ANGLE:",Rleg_angle)
    # print("LLEG ANGLE:",Lleg_angle)
    #
    # print("NECK_LSHOULDER ANGLE:",Lshoulder_angle)
    # print("NECK_RSHOULDER ANGLE:",Rshoulder_angle)
    # print(Rshoulder_angle+Lshoulder_angle)

    if (
        abs(Rarm_angle - 180) <= 10
        and abs(Larm_angle - 180) <= 10
        and abs(Rleg_angle - 180) <= 10
        and abs(Lleg_angle - 180) <= 10
        and abs((Rshoulder_angle+Lshoulder_angle) - 180) <= 14
    ):
        return "T POSE RECOGNIZED"
    else:
        return "NOT T POSE"

def read_skeletons(file_name):
    with open('../body_data/second_attempt/'+file_name+'MOCAP.json', 'r') as f:
        data = json.load(f)

    keypoints = [[]]
    for i,frame in enumerate(data):
        keypoints.append([])
        for joint in frame['keypoints']:
            keypoints[i].append(joint['Position'])

    keypoints=np.array(keypoints)

    return keypoints


def main():

    file_name=''
    frame=0

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        #frame = int(sys.argv[2])
    else:
        print("No file name provided.")
        exit(1)

    skeletons=read_skeletons(file_name)
    # skeleton=skeletons[frame]
    #
    # # split the points into x, y, z coordinates
    # x = [p[0] for p in skeleton]
    # y = [p[1] for p in skeleton]
    # z = [p[2] for p in skeleton]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111) #, projection='3d')
    #
    # ax.scatter(z,y) #,z)
    #
    # for bone, indices in bones.items():
    #     idx1, idx2 = indices
    #     ax.plot([z[idx1], z[idx2]], [y[idx1], y[idx2]], color='red')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # #ax.set_zlabel('Z')
    #ax.set_title(f'Frame {frame}')

    #THESE ARE ANGLES IN THE PLANE ZY (INSTEAD OF XY)
    #skeleton[bones["Rarm"][0]][2] -> THE Z COORD OF THE FIRST POINT OF THE R ARM BONE
    #skeleton[bones["Rarm"][1]][1] -> THE Y COORD OF THE SECOND POINT OF THE R ARM BONE

    for i,skeleton in enumerate(skeletons):
        if (skeleton!=[]):
            print(f"---------FRAME {i}---------")
            Rarm_angle=compute_angle(skeleton[bones["Rarm"][0]][2],skeleton[bones["Rarm"][0]][1],skeleton[bones["Rarm"][1]][2],skeleton[bones["Rarm"][1]][1],skeleton[bones["Rforearm"][0]][2],skeleton[bones["Rforearm"][0]][1],skeleton[bones["Rforearm"][1]][2],skeleton[bones["Rforearm"][1]][1])
            Larm_angle=compute_angle(skeleton[bones["Larm"][0]][2],skeleton[bones["Larm"][0]][1],skeleton[bones["Larm"][1]][2],skeleton[bones["Larm"][1]][1],skeleton[bones["Lforearm"][0]][2],skeleton[bones["Lforearm"][0]][1],skeleton[bones["Lforearm"][1]][2],skeleton[bones["Lforearm"][1]][1])

            Rleg_angle=compute_angle(skeleton[bones["Rthigh"][0]][2],skeleton[bones["Rthigh"][0]][1],skeleton[bones["Rthigh"][1]][2],skeleton[bones["Rthigh"][1]][1],skeleton[bones["Rshin"][0]][2],skeleton[bones["Rshin"][0]][1],skeleton[bones["Rshin"][1]][2],skeleton[bones["Rshin"][1]][1])
            Lleg_angle=compute_angle(skeleton[bones["Lthigh"][0]][2],skeleton[bones["Lthigh"][0]][1],skeleton[bones["Lthigh"][1]][2],skeleton[bones["Lthigh"][1]][1],skeleton[bones["Lshin"][0]][2],skeleton[bones["Lshin"][0]][1],skeleton[bones["Lshin"][1]][2],skeleton[bones["Lshin"][1]][1])

            Rshoulder_angle=compute_angle(skeleton[bones["Rshoulder"][0]][2],skeleton[bones["Rshoulder"][0]][1],skeleton[bones["Rshoulder"][1]][2],skeleton[bones["Rshoulder"][1]][1],skeleton[bones["neck"][0]][2],skeleton[bones["neck"][0]][1],skeleton[bones["neck"][1]][2],skeleton[bones["neck"][1]][1])
            Lshoulder_angle=compute_angle(skeleton[bones["Lshoulder"][0]][2],skeleton[bones["Lshoulder"][0]][1],skeleton[bones["Lshoulder"][1]][2],skeleton[bones["Lshoulder"][1]][1],skeleton[bones["neck"][0]][2],skeleton[bones["neck"][0]][1],skeleton[bones["neck"][1]][2],skeleton[bones["neck"][1]][1])
            print(compute_T_pose(Rarm_angle,Larm_angle,Rleg_angle,Lleg_angle,Rshoulder_angle,Lshoulder_angle))


    #plt.show()


if __name__ == '__main__':
    main()
