import matplotlib.pyplot as plt
import numpy as np
import sys

# Define a series of vectors
import json

# load the JSON file
with open('../body_data/bodies0.json', 'r') as f:
    data = json.load(f)

# Extract the "keypoint" vectors
keypoints = []
for body in data.values():
    for body_part in body['body_list']:
        keypoints.append(body_part['keypoint'])
keypoints = np.array(keypoints)

print(keypoints.shape)

# create a list of 3D vectors
vectors = []
for frame in keypoints:
    frame_vectors = []
    for point in frame:
        x, y, z = point[0], point[1], point[2]
        frame_vectors.append([x, y, z])
    vectors.append(frame_vectors)

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)
ax.view_init(azim=-90, elev=90)

ax.invert_yaxis()
ax.set_xlim([-1, 1])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1, 1])


print("Generating plots for each frame...")

completed_images=0

# Loop through the vectors and plot each one
for i, vector in enumerate(vectors):
    if i > 0:
        ax.collections[0].remove()  # Remove the scatter plot from the previous frame
    vector_array = np.array(vector)
    ax.scatter(vector_array[:, 0], vector_array[:, 1], vector_array[:, 2], c=range(len(vector_array)), cmap='rainbow')
    ax.set_title(f'Frame {i}')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Save the plot as an image
    plt.savefig(f'frame_{i}.png')

    completed_images += 1
    percentage = int(completed_images / len(vectors) * 100)
    sys.stdout.write(f"\rPlotting data: {percentage}%")
    sys.stdout.flush()

# Create the video from the frames
import cv2
import os

print("")
print("Building video...")

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

os.system('ffmpeg -framerate 60 -i frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p output3D.mp4 -y')

import glob

# Remove all the files with pattern 'frame_*.png'
print("Removing frames")
for file in glob.glob('frame_*.png'):
    os.remove(file)
