import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes
import matplotlib.colors as mcolors
import sys


index = 0
colors = [ 'darkred', 'red']

file_name = ''
## which sample
if len(sys.argv) != 2:
    print("Not enough arguments")
    exit(0)
else:
    sample = int(sys.argv[1])  # Assuming the argument is an integer

if sample == 0:
    print("Reference")
    file_name = 'groundTruthZED'
    colors = [ 'blue', 'cyan']
    start = 1
    end = 19
elif sample == 1:
    print("Sample 1")
    file_name = 'groundTruthZED'
    start = 21
    end = 39
elif sample == 2:
    print("Sample 2")
    file_name = 'groundTruthZED'
    start = 41
    end = 59
elif sample == 3:
    print("Sample 3")
    file_name = 'sampleZED'
    start = 1
    end = 19
elif sample == 4:
    print("Sample 4")
    file_name = 'sampleZED'
    start = 21
    end = 39
elif sample == 5:
    print("Sample 5")
    file_name = 'sampleZED'
    start = 41
    end = 59

############## taking 20 points for squat sample
pose_index_sample = []
pose_index_sample, skeletons_sample = ZED_alignment.main(file_name)
# print("pose_index_sample: ", pose_index_sample)

############## JOINT pelvis  sample
plot_pelvis_sample = []
for i,skeleton in enumerate(skeletons_sample):
    if i > pose_index_sample[start] and i < pose_index_sample[end]: # first squat
        plot_pelvis_sample.append(skeleton[index])

plot_pelvis_sample = np.array(plot_pelvis_sample)
# print("plot_pelvis_sample:",plot_pelvis_sample)

##################################################### PLOT
fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111)
print(plot_pelvis_sample)
plot_pelvis_sample -= plot_pelvis_sample[0]
print(plot_pelvis_sample)

y_min_gt = np.min(plot_pelvis_sample)
y_max_gt = np.max(plot_pelvis_sample)
bounds = [y_min_gt, y_max_gt]
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
norm = mcolors.Normalize(vmin=y_min_gt, vmax=y_max_gt)

ax.scatter(plot_pelvis_sample[:, 0], plot_pelvis_sample[:, 1],  c=plot_pelvis_sample[:, 1], cmap=cmap, norm=norm, label='SAMPLE', s=2)


### vline
points_sample = np.array(plot_pelvis_sample)

# Find the index of the point with the minimum y-coordinate
min_y_index_sample = np.argmin(points_sample[:, 1])
point_with_min_y_sample = points_sample[min_y_index_sample]

# Print the minimum point
print("Minimum y:", point_with_min_y_sample)
ax.plot([0.0,0.0], [point_with_min_y_sample[1], 0.0], c='gray', linestyle='--')
plt.axis('equal')
# plt.show()
plt.savefig('../ZEDresults/PELVIS/pelvis_ZED_sample'+str(sample)+'.png')
