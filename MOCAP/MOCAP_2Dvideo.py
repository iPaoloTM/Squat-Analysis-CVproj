import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import glob
import sys

def read_skeletons(file_name):

    with open('../body_data/'+file_to_read+'.json', 'r') as f:
        data = json.load(f)

    # Extract the "keypoint" vectors
    keypoints = [[]]
    for i,frame in enumerate(data):
        keypoints.append([])
        for joint in frame['keypoints']:
            keypoints[i].append(joint['Position'])

    return np.array(keypoints)

def main():

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        #print("First argument:", first_argument)
    else:
        print("No file name provided.")
        exit(1)

    keypoints = read_skeletons(file_name)

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

    print("Generating plots for each frame...")

    completed_images=0

    # Loop through the vectors and plot each one
    for i, vector in enumerate(vectors):
        if i > 0:
            ax.collections[0].remove()  # Remove the scatter plot from the previous frame
        vector_array = np.array(vector)
        ax.scatter(vector_array[:, 0], vector_array[:, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title(f'Frame {i}')
        # Save the plot as an image
        plt.savefig(f'frame_{i}.png')
        #print(f"Plot {i} saved")

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

    os.system('ffmpeg -framerate 60 -i frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+file_name+'.mp4 -y')

    # Remove all the files with pattern 'frame_*.png'
    print("Removing frames")
    for file in glob.glob('frame_*.png'):
        os.remove(file)

    os.system('open output2D_'+file_name+'.mp4')

if __name__ == '__main__':
    main()
