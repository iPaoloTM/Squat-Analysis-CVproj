import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm
import open3d as o3d
import MOCAP_alignment


pose_index_sample = MOCAP_alignment.main('sampleMOCAP')
pelvises_sample = MOCAP_alignment.read_joint('sampleMOCAP', 'Aliprandi_Girardi:Hip', pose_index_sample[0], pose_index_sample[1])

pose_index_gt = MOCAP_alignment.main('groundTruthMOCAP')
pelvises_gt = MOCAP_alignment.read_joint('groundTruthMOCAP', 'Aliprandi_Girardi:Hip', pose_index_gt[0], pose_index_gt[1])

fig_pelvis = plt.figure()
ax = fig_pelvis.add_subplot(111, projection='3d')
ax.set_title("PELVIS FROM MOCAP")
ax.plot(pelvises_sample[:, 2], pelvises_sample[:, 1], pelvises_sample[:, 0], c='red', label='SAMPLE')
ax.plot(pelvises_gt[:, 2], pelvises_gt[:, 1], pelvises_gt[:, 0], c='blue', label='GT')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_ylabel('Z Label')

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

# Set legend properties
legend = ax.legend(title='Points', loc='upper right')
plt.show()

#####################
