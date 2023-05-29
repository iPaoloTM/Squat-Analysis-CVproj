import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def isBetween(var, left_extr, right_extr):
    if var >= left_extr and var <= right_extr:
        return True
    else:
        return False

# Define squat phases
STANDING = 0
GOING_DOWN = 1
DEEP_SQUAT = 2
GOING_UP = 3

joint_names34 = {
    0: 'PELVIS',
    1: 'NAVAL_SPINE',
    2: 'CHEST_SPINE',
    3: 'NECK',
    4: 'LEFT_CLAVICLE',
    5: 'LEFT_SHOULDER',
    6: 'LEFT_ELBOW',
    7: 'LEFT_WRIST',
    8: 'LEFT_HAND',
    9: 'LEFT_HANDTIP',
    10: 'LEFT_THUMB',
    11: 'RIGHT_CLAVICLE',
    12: 'RIGHT_SHOULDER',
    13: 'RIGHT_ELBOW',
    14: 'RIGHT_WRIST',
    15: 'RIGHT_HAND',
    16: 'RIGHT_HANDTIP',
    17: 'RIGHT_THUMB',
    18: 'LEFT_HIP',
    19: 'LEFT_KNEE',
    20: 'LEFT_ANKLE',
    21: 'LEFT_FOOT',
    22: 'RIGHT_HIP',
    23: 'RIGHT_KNEE',
    24: 'RIGHT_ANKLE',
    25: 'RIGHT_FOOT',
    26: 'HEAD',
    27: 'NOSE',
    28: 'LEFT_EYE',
    29: 'LEFT_EAR',
    30: 'RIGHT_EYE',
    31: 'RIGHT_EAR',
    32: 'LEFT_HEEL',
    33: 'RIGHT_HEEL'
}


joint_numbers34 = {v: k for k, v in joint_names34.items()}

file_to_read = 'Squat2'

# Load the JSON file
with open('../body_data/first_attempt/'+file_to_read+'.json', 'r') as f:
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

keypoints = np.array(keypoints)

print(keypoints.shape)

# Create a list of 2D vectors
vectors = []
for frame in keypoints:
    frame_vectors = []
    for point in frame:
        x, y = point[0], point[1]
        frame_vectors.append([x, y])
    vectors.append(frame_vectors)

# Set up the plot
fig, ax = plt.subplots()
ax.invert_yaxis()

##########################################################

print("\033[93mComputing pose for each frame...\033[0m")

completed_images=0

# Initialize variables for squat tracking
squat_phase = STANDING


# Loop through the vectors and plot each one
for i, vector in enumerate(vectors):
    if i > 0:
        ax.collections[0].remove()  # Remove the scatter plot from the previous frame

    vector_array = np.array(vector)
    ax.scatter(vector_array[:, 0], vector_array[:, 1], c=range(len(vector_array)), cmap='rainbow')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    # Save the plot as an image
    plt.savefig(f'frame_{i}.png')
    #print(f"Plot {i} saved")

    completed_images += 1
    percentage = int(completed_images / len(vectors) * 100)
    color = ''
    if percentage < 33:
        color='\033[31m'
    elif percentage < 66:
        color='\033[93m'
    else:
        color='\033[92m'

    sys.stdout.write(f"\r \033[93mPlotting data: "+color+f"{percentage}% \033[0m")
    sys.stdout.flush()
    #ax.set_title(f'Frame {i}-SQUAT: {squat_phase}')

    if i<len(vectors)-1:
        # Get position of key body parts
        pelvis_pos = vector[joint_numbers34['PELVIS']]
        left_hip_pos = vector[joint_numbers34['LEFT_HIP']]
        left_knee_pos = vector[joint_numbers34['LEFT_KNEE']]
        left_ankle_pos = vector[joint_numbers34['LEFT_ANKLE']]
        right_hip_pos = vector[joint_numbers34['RIGHT_HIP']]
        right_knee_pos = vector[joint_numbers34['RIGHT_KNEE']]
        right_ankle_pos = vector[joint_numbers34['RIGHT_ANKLE']]

        next_pelvis_pos = vectors[i+1][joint_numbers34['PELVIS']]
        next_left_hip_pos = vectors[i+1][joint_numbers34['LEFT_HIP']]
        next_left_knee_pos = vectors[i+1][joint_numbers34['LEFT_KNEE']]
        next_left_ankle_pos = vectors[i+1][joint_numbers34['LEFT_ANKLE']]
        next_right_hip_pos = vectors[i+1][joint_numbers34['RIGHT_HIP']]
        next_right_knee_pos = vectors[i+1][joint_numbers34['RIGHT_KNEE']]
        next_right_ankle_pos = vectors[i+1][joint_numbers34['RIGHT_ANKLE']]


        if squat_phase != GOING_DOWN:
            if isBetween(left_hip_pos[1]-next_left_hip_pos[1],-0.3,0.3) or isBetween(right_hip_pos[1]-next_right_hip_pos[1],-0.3,0.3):
                squat_phase = STANDING
                ax.set_title(f'Frame {i}-SQUAT: STANDING',color='black')
            elif left_hip_pos[1] > next_left_hip_pos[1] or right_hip_pos[1] > next_right_hip_pos[1]:
                squat_phase = GOING_UP
                ax.set_title(f'Frame {i}-SQUAT: GOING UP',color='green')
            elif left_hip_pos[1] < next_left_hip_pos[1] or right_hip_pos[1] < next_right_hip_pos[1]:
                squat_phase = GOING_DOWN
                ax.set_title(f'Frame {i}-SQUAT: GOING DOWN',color='blue')
        else:
            if isBetween(pelvis_pos[1]-left_knee_pos[1],-20,20) and isBetween(pelvis_pos[1]-right_hip_pos[1],-20,20):
                squat_phase = DEEP_SQUAT
                ax.set_title(f'Frame {i}-SQUAT: DEEP SQUAT',color='purple')



# Create the video from the frames
import cv2
import os

print("")
print("\033[93mBuilding video...\033[0m")

img_array = []
for i in range(len(vectors)):
    img = cv2.imread(f'frame_{i}.png')
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
#
# out.release()

os.system('ffmpeg -framerate 60 -i frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+file_to_read+'_compute-pose.mp4 -y')

import glob

# Remove all the files with pattern 'frame_*.png'
print("\033[93mRemoving frames\033[0m")
for file in glob.glob('frame_*.png'):
    os.remove(file)
