import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes
import ZED_3Dplot

##################### sampleZED

############## SPATIAL ALIGNMENT
## get skeletons aligned
centered_skeletons = []
skeletons = ZED_alignment.read_skeletons('groundTruthZED')
for joint in skeletons:
    if len(joint)!=0:
        skeleton = ZED_3Dplot.center_skeleton(np.array(joint))
        centered_skeletons.append(skeleton)

############## TEMPORAL ALIGNMENT
pose_index_sample = ZED_alignment.main(centered_skeletons)

############## JOINT HEAD sample vs groundTruth
index = 26
list_of_head_coord = []
for i,skeleton in enumerate(centered_skeletons):
    if i > pose_index_sample[0] and i < pose_index_sample[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)

##################################################### PLOT
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
ax.set_title("head FROM ZED")

# Add a vertical line
vertical_line_position = np.mean(list_of_head_coord[:,0])
ax.scatter(list_of_head_coord[:, 0], list_of_head_coord[:, 1], c='red', label='SAMPLE')
ax.axvline(x=vertical_line_position, color='green', linestyle='-', linewidth=5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.axis('equal')
# plt.xlim(0, 1)
# plt.ylim(0, 1.75)


# Set legend properties
legend = ax.legend(title='Points', loc='upper right')

########### PLOT from upper
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
ax.set_title("head from upper FROM ZED")

# Define the number of concentric circles
num_circles = 3

# Set the radius of the outermost circle
outer_radius = 1.0

# Set the center coordinates
center_x = 0
center_y = 0

# Define the colors for each area
colors = [[1.0, 0.6, 0.6], [1.0, 0.8, 0.6], [0.7, 0.9, 0.7]]

# Plot the concentric circles
for i in range(num_circles):
    radius = outer_radius * (num_circles - i) / num_circles
    color = colors[i % len(colors)]  # Cycle through the colors
    circle = plt.Circle((center_x, center_y), radius, color=color, fill=True)
    ax.add_patch(circle)

# Set the aspect ratio to be equal
ax.set_aspect('equal')

# Set the limits of the plot
ax.set_xlim(center_x - outer_radius, center_x + outer_radius)
ax.set_ylim(center_y - outer_radius, center_y + outer_radius)

ax.scatter(list_of_head_coord[:, 0], list_of_head_coord[:, 1], c='red', label='SAMPLE')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.axis('equal')
plt.gca().invert_yaxis() # Invert y-axis to represent the movement from the top

# Set legend properties
legend = ax.legend(title='Points', loc='upper right')
plt.show()
