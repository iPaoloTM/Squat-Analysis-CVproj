from h36m_skeleton import H36mSkeleton
from scheletroZED import Poses_3d
import numpy as np
import sys
import os


# Check if there is command-line argument input_file
if len(sys.argv) != 2:
    print("Invalid number of arguments. Please provide exactly one input.")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file + ".bvh"

if os.path.exists(output_file):
    print("Output file already exists.")
else:
    print("Creating output file:", output_file)

# Create an instance of the Skeleton class
skeleton = H36mSkeleton()
poses = Poses_3d()
poses_3d = poses.get_poses(input_file=input_file)
# Access the skeleton properties
print(skeleton.root)
print(skeleton.keypoint2index)
print(skeleton.children)
print(poses_3d)
print(input_file)

# Convert poses_3d to a NumPy array
poses_3d = np.array(poses_3d)
output_file = input_file + ".bvh"
# Call the poses2bvh method with the correct arguments
channels, header = skeleton.poses2bvh(poses_3d, output_file=output_file)
