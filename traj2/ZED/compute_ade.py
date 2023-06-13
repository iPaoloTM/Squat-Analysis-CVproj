import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import ZED_alignment
from scipy.spatial import procrustes
import matplotlib.colors as mcolors
import sys

def compute_ade(list1, list2):

    assert len(list1) == len(list2), "Point lists must have the same length."
    distances = np.linalg.norm(list1 - list2, axis=1)
    ade = np.mean(distances)

    return ade, distances



# index = 27
index = 0
############### READ reference

pose_index_gt = []
pose_index_gt, skeletons_gt = ZED_alignment.main('groundTruthZED')

plot_pelvis_gt = []
for i,skeleton in enumerate(skeletons_gt):
    if i > pose_index_gt[1] and i < pose_index_gt[19]: # first squat
        plot_pelvis_gt.append(skeleton[index])

############# important keypoints
interesting_pelvis_coor_gt = []
for i,skeleton in enumerate(pose_index_gt):
    if i > 1 and i < 19:
        interesting_pelvis_coor_gt.append(skeletons_gt[pose_index_gt[i]][index])

interesting_pelvis_coor_gt = np.array(interesting_pelvis_coor_gt)
interesting_pelvis_coor_gt -= interesting_pelvis_coor_gt[0]


# print("interesting_pelvis_coor_gt: ",interesting_pelvis_coor_gt)
# print("len: ", len(interesting_pelvis_coor_gt))


file_name = ''
## which sample
if len(sys.argv) != 2:
    print("Not enough arguments")
    exit(0)
else:
    sample = int(sys.argv[1])  # Assuming the argument is an integer


if sample == 1:
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


############## JOINT pelvis salienti
interesting_pelvis_coor_sample = []
for i,skeleton in enumerate(pose_index_sample):
    if i > start and i < end:
        interesting_pelvis_coor_sample.append(skeletons_sample[pose_index_sample[i]][index])

interesting_pelvis_coor_sample = np.array(interesting_pelvis_coor_sample)
interesting_pelvis_coor_sample -= interesting_pelvis_coor_sample[0]

# print("interesting_pelvis_coor_sample: ",interesting_pelvis_coor_sample)
# print("len: ", len(interesting_pelvis_coor_sample))

################# plot
fig_pelvis_traj = plt.figure()
ax = fig_pelvis_traj.add_subplot(111)

ax.scatter(interesting_pelvis_coor_gt[:, 0], interesting_pelvis_coor_gt[:, 1],c='blue', label='GROUDTRUTH', s=3)
ax.scatter(interesting_pelvis_coor_sample[:, 0], interesting_pelvis_coor_sample[:, 1],c='red', label='SAMPLE', s=3)

plt.axis('equal')
# plt.show()

plt.savefig('../ZEDresults/PELVIS/pelvis_ZED_groundTruthVSsample_sampledTraj_'+str(sample)+'.png')

############### computing ADE


### vline
points_sample = np.array(plot_pelvis_sample)
points_gt = np.array(plot_pelvis_gt)

# Find the index of the point with the minimum y-coordinate
min_y_index_sample = np.argmin(points_sample[:, 1])
point_with_min_y_sample = points_sample[min_y_index_sample]

min_y_index_gt = np.argmin(points_gt[:, 1])
point_with_min_y_gt = points_gt[min_y_index_gt]
# print("Point with the minimum y-coordinate:", point_with_min_y_sample, point_with_min_y_gt)

if point_with_min_y_sample[1] < point_with_min_y_gt[1]:
    min_point = point_with_min_y_sample
else:
    min_point = point_with_min_y_gt

x_values = np.linspace(0.0, 0.0, len(interesting_pelvis_coor_gt))
y_values = np.linspace(min_point[1], 0.0, len(interesting_pelvis_coor_gt))
z_values = np.linspace(0.0, 0.0, len(interesting_pelvis_coor_gt))
points_array = np.column_stack((x_values, y_values, z_values))


plt.axis('equal')

ade, distances = compute_ade(interesting_pelvis_coor_gt, interesting_pelvis_coor_sample)
print("ADE:", ade)
# print("Current distances:", distances)

ade_gt, distances = compute_ade(interesting_pelvis_coor_gt, points_array)
print("ADE gt respect to verticalLine:", ade_gt)
# print("Current distances:", distances)

ade_sample, distances = compute_ade(interesting_pelvis_coor_sample, points_array)
print("ADE SAMPLE "+str(sample)+" respect to verticalLine:", ade_sample)
# print("Current distances:", distances)

# plot of ADE
time = range(len(distances))

fig_ade = plt.figure()
ax = fig_ade.add_subplot(111)
plt.plot(time, distances, marker='o')  # Create a line plot
plt.xlabel('Time')  # Set x-axis label
plt.ylabel('ADE')  # Set y-axis label
# plt.show()
plt.savefig('../ZEDresults/PELVIS/pelvis_ZED_groundTruthVSsample_sampledTraj_Distances_'+str(sample)+'.png')
