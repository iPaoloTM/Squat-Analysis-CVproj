import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
from scipy.spatial import procrustes
import matplotlib.colors as mcolors
import sys


index = 0
############### READ reference

pose_index_gt = []
pose_index_gt, skeletons_gt = MOCAP_alignment.main('groundTruthMOCAP')

plot_pelvis_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[1] and i < pose_index_gt[19]: # first squat
        plot_pelvis_gt.append(skeleton[index])



file_name = ''
## which sample
if len(sys.argv) != 2:
    print("Not enough arguments")
    exit(0)
else:
    sample = int(sys.argv[1])  # Assuming the argument is an integer


if sample == 1:
    print("Sample 1")
    file_name = 'groundTruthMOCAP'
    start = 21
    end = 39
elif sample == 2:
    print("Sample 2")
    file_name = 'groundTruthMOCAP'
    start = 41
    end = 59
elif sample == 3:
    print("Sample 3")
    file_name = 'sampleMOCAP'
    start = 1
    end = 19
elif sample == 4:
    print("Sample 4")
    file_name = 'sampleMOCAP'
    start = 21
    end = 39
elif sample == 5:
    print("Sample 5")
    file_name = 'sampleMOCAP'
    start = 41
    end = 59

############## taking 20 points for squat sample
pose_index_sample = []
pose_index_sample, skeletons_sample = MOCAP_alignment.main(file_name)
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

colors = [ 'darkred', 'red']
y_min_gt = np.min(plot_pelvis_sample)
y_max_gt = np.max(plot_pelvis_sample)
bounds = [y_min_gt, y_max_gt]
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
norm = mcolors.Normalize(vmin=y_min_gt, vmax=y_max_gt)
ax.scatter(plot_pelvis_sample[:, 2], plot_pelvis_sample[:, 1],  c=plot_pelvis_sample[:, 1], cmap=cmap, norm=norm, label='SAMPLE', s=2)


### gt

plot_pelvis_gt = np.array(plot_pelvis_gt)
plot_pelvis_gt -= plot_pelvis_gt[0]

colors_2 = [ 'darkblue', 'cyan']
y_min = np.min(plot_pelvis_gt)
y_max = np.max(plot_pelvis_gt)
bounds = [y_min, y_max]
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors_2)
norm = mcolors.Normalize(vmin=y_min, vmax=y_max)
plt.scatter(plot_pelvis_gt[:, 2], plot_pelvis_gt[:, 1], c=plot_pelvis_gt[:, 1], cmap=cmap, norm=norm,  label='GROUDTRUTH', s=3)
# cbar = plt.colorbar()

### vline
points_sample = np.array(plot_pelvis_sample)
points_gt = np.array(plot_pelvis_gt)

# Find the index of the point with the minimum y-coordinate
min_y_index_sample = np.argmin(points_sample[:, 1])
point_with_min_y_sample = points_sample[min_y_index_sample]

min_y_index_gt = np.argmin(points_gt[:, 1])
point_with_min_y_gt = points_gt[min_y_index_gt]
print("Point with the minimum y-coordinate:", point_with_min_y_sample, point_with_min_y_gt)

if point_with_min_y_sample[1] < point_with_min_y_gt[1]:
    min_point = point_with_min_y_sample
else:
    min_point = point_with_min_y_gt

# Print the minimum point
print("Minimum y:", min_point)
ax.plot([0.0,0.0], [min_point[1], 0.0], c='gray', linestyle='--')
plt.axis('equal')
# plt.show()
# exit(0)
plt.savefig('../MOCAP_results/pelvis_MOCAP_groundTruthVSsample'+str(sample)+'.png')
