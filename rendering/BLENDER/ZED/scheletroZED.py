import json
import matplotlib.pyplot as plt
import numpy as np
import sys

class Poses_3d:
    def get_poses(self, input_file):
        # Load the JSON file
        with open('/Users/letiziagirardi/Desktop/UNIVERSITY/MAGISTRALE/SEMESTRE_2/Squat-Analysis-CVproj/body_data/'+input_file+'.json', 'r') as f:
            data = json.load(f)

        # Extract the "keypoint" vectors
        keypoints = []
        for body in data.values():
            for body_part in body['body_list']:
                keypoints.append(body_part['keypoint'])
        keypoints = np.array(keypoints)

        # create a list of 3D vectors
        vectors = []
        for frame in keypoints:
            frame_vectors = []
            for point in frame:
                x, y, z = point[0], point[1], point[2]
                frame_vectors.append([x, y, z])
            vectors.append(frame_vectors)

        # Loop through the vectors
        for i, vector in enumerate(vectors):
            vector_array = np.array(vector)

        return vectors
