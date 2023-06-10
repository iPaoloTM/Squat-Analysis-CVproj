import matplotlib.pyplot as plt
import numpy as np
import json
import sys
##################################################################################################
#FA LA STESSA COSA DI COMPUTE_POSE.PY MA STAMPA SOLTANTO LE FASI A SCHERMO, CON LE COORDINATE
##################################################################################################
# Define keypoint indices for squat tracking
# PELVIS = 0
# LEFT_HIP = 18
# LEFT_KNEE = 19
# LEFT_ANKLE = 20
# RIGHT_HIP = 22
# RIGHT_KNEE = 23
# RIGHT_ANKLE = 24
#
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


##########################################################

file_to_read = 'Squat1'

# Load the JSON file
with open('body_data/first_attempt/'+file_to_read+'.json', 'r') as f:
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

squat_phase = STANDING
last_left_knee_pos = None
last_right_knee_pos = None


# Create a list of 2D vectors
vectors = []
for frame in keypoints:
    frame_vectors = []
    for point in frame:
        x, y = point[0], point[1]
        frame_vectors.append([x, y])
    vectors.append(frame_vectors)

lowest_point=0

for i, vector in enumerate(vectors):
    if i<len(vectors)-1:
        print(f'#Frame number:{i}')

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

        #option+shift+8=
        # print(pelvis_pos[1])
        # print(left_hip_pos[1])
        # print(left_knee_pos[1])
        # print(left_ankle_pos[1])
        # print(right_hip_pos[1])
        # print(right_knee_pos[1])
        # print(right_ankle_pos[1])

        #print(f"\033[31mRIGHT HIP POS DIFFERENCE:\033[0m {right_hip_pos[1]-next_right_hip_pos[1]}")
        #print(f"\033[92mLEFT HIP POS DIFFERENCE:\033[0m {left_hip_pos[1]-next_left_hip_pos[1]}")


        if squat_phase == STANDING:
            if left_hip_pos[1] > next_left_hip_pos[1] and right_hip_pos[1] > next_right_hip_pos[1]:
                squat_phase = GOING_DOWN
            print("\033[31m--------SQUAT: GOING DOWN--------\033[0m")
        elif squat_phase == GOING_DOWN:
            if left_hip_pos[1] < next_left_hip_pos[1] and right_hip_pos[1] < next_right_hip_pos[1]:
                squat_phase = GOING_UP
            print("\033[92m--------SQUAT: GOING UP--------\033[0m")
        elif squat_phase == GOING_UP:
            if left_hip_pos[1] >= next_left_hip_pos[1] and right_knee_pos[1] >=  next_right_hip_pos[1]:
                squat_phase = STANDING
            print("--------SQUAT: STANDING--------")

        lowest_point=max(lowest_point,pelvis_pos[1])
        print(lowest_point)
