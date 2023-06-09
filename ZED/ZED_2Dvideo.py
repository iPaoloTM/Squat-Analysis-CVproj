import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys
import cv2
import os

bones={"pelvis+abs": [0,1], "chest": [1,2], "neck": [3,26],
       "Rclavicle":[3,11],"Rshoulder":[11,12],"Rarm":[12,13], "Rforearm":[13,14],
       "chest1":[2,11],"chest2":[2,3],"chest3":[2,4],
       "Lclavicle":[3,4],"Lshoulder":[4,5], "Larm":[5,6], "Lforearm":[6,7],
       "Rhip":[0,22], "Rthigh":[22,23],"Rshin":[23,24],
       "Lhip":[0,18], "Lthigh":[18,19],"Lshin":[19,20],
       "Rfoot":[25,33],"Rankle":[24,33],"Lfoot":[21,32],"Lankle":[20,32]}

def read_skeleton(file_name):

    # Load the JSON file
    with open('../body_data/'+file_name+'.json', 'r') as f:
        data = json.load(f)

    tracking_state = []
    action_state = []
    # Extract the "keypoint" vectors
    keypoints = []
    for body in data.values():
        for body_part in body['body_list']:
            keypoints.append(body_part['keypoint_2d'])
            tracking_state.append(body_part['tracking_state'])
            action_state.append(body_part['action_state'])

    return np.array(keypoints),tracking_state,action_state

def plot_skeletons(vectors,tracking_state,action_state):

    fig, ax = plt.subplots()
    ax.invert_yaxis()

    print("Generating plots for each frame...")

    completed_images=0

    # Loop through the vectors and plot each one
    for i, vector in enumerate(vectors):
        if i > 0:
            ax.collections[0].remove()  # Remove the scatter plot from the previous frame
        vector_array = np.array(vector)

        ax.scatter(vector_array[:, 0], vector_array[:, 1], color='orange')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-1.5, 0.8])
        ax.set_title(f'Frame {i}-'+tracking_state[i]+'-'+action_state[i])
        # Save the plot as an image
        plt.savefig(f'frame_{i}.png')
        #print(f"Plot {i} saved")

        completed_images += 1
        percentage = int(completed_images / len(vectors) * 100)
        sys.stdout.write(f"\rPlotting data: {percentage}%")
        sys.stdout.flush()

def main():

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        print("No file name provided.")
        exit(1)

    keypoints,tracking_state,action_state=read_skeleton(file_name)

    # Create a list of 2D vectors
    vectors = []
    for frame in keypoints:
        frame_vectors = []
        for point in frame:
            x, y = point[0], point[1]
            frame_vectors.append([x, y])
        vectors.append(frame_vectors)

    plot_skeletons(vectors,tracking_state,action_state)

    print("")
    print("Building video...")

    img_array = []
    for i in range(len(vectors)):
        img = cv2.imread(f'frame_{i}.png')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    os.system('ffmpeg -framerate 60 -i frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+file_name+'.mp4 -y')

    # Remove all the files with pattern 'frame_*.png'
    print("Removing frames")
    for file in glob.glob('frame_*.png'):
        os.remove(file)

    os.system('open output2D_'+file_name+'ZED.mp4')

if __name__ == '__main__':
    main()
