import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment
from scipy.spatial import procrustes
import MOCAP_3Dplot

##################### sampleMOCAP

############## SPATIAL ALIGNMENT
## get skeletons aligned
centered_skeletons = []
skeletons = MOCAP_alignment.read_skeletons('sampleMOCAP')
for joint in skeletons:
    if len(joint) > 18 and len(joint)!=0:
        skeleton = MOCAP_3Dplot.center_skeleton(np.array(joint))
        centered_skeletons.append(skeleton)


############## TEMPORAL ALIGNMENT
pose_index_sample = MOCAP_alignment.main(centered_skeletons)

############## JOINT HEAD sample vs groundTruth
index = 0
list_of_head_coord = []
for i,skeleton in enumerate(centered_skeletons):
    if i > pose_index_sample[0] and i < pose_index_sample[1]:
        list_of_head_coord.append(skeleton[index])

list_of_head_coord = np.array(list_of_head_coord)

########### PLOT
fig_head = plt.figure()
ax = fig_head.add_subplot(111)
ax.set_title("head FROM MOCAP")
ax.scatter(list_of_head_coord[:, 0], list_of_head_coord[:, 1], c='red', label='SAMPLE')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.axis('equal')
# Set legend properties
legend = ax.legend(title='Points', loc='upper right')
plt.show()
