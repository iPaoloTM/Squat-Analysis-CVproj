import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import sys
import json


bones={"pelvis+abs": [0,1], "chest": [1,2], "neck": [3,26],
       "Rclavicle":[3,11],"Rshoulder":[11,12],"Rarm":[12,13], "Rforearm":[13,14],
       "chest1":[2,11],"chest2":[2,3],"chest3":[2,4],
       "Lclavicle":[3,4],"Lshoulder":[4,5], "Larm":[5,6], "Lforearm":[6,7],
       "Rhip":[0,22], "Rthigh":[22,23],"Rshin":[23,24],
       "Lhip":[0,18], "Lthigh":[18,19],"Lshin":[19,20],
       "Rfoot":[25,33],"Rankle":[24,33],"Lfoot":[21,32],"Lankle":[20,32]}

def read_skeleton(file_name, frame):
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    keyskeleton = []
    for i,body in enumerate(data.values()):
        if i==int(frame):
            for body_part in body['body_list']:
                keyskeleton.append(body_part['keypoint_2d'])

    return np.array(keyskeleton[0])

def plot_skeleton(skeleton):

    x = [point[0] for point in skeleton] 
    y = [point[1] for point in skeleton]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y,marker='o', color='orange')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.invert_yaxis()

    for bone, indices in bones.items():
        idx1, idx2 = indices
        ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], color='orange')

    plt.show()

def main():

    if len(sys.argv) > 2:
        file_name = sys.argv[1]
        frame = int(sys.argv[2])
    else:
        print("Not enough arguments")
        exit(1)

    keyskeleton=read_skeleton(file_name,frame)

    plot_skeleton(keyskeleton)



if __name__ == '__main__':
    main()
