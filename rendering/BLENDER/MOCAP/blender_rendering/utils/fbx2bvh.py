import bpy
import numpy as np
import sys
from os import listdir, path

def fbx2bvh(data_path, file):
    sourcepath = data_path+"/"+file
    bvh_path = data_path+"/"+file.split(".fbx")[0]+".bvh"

    bpy.ops.import_scene.fbx(filepath=sourcepath)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if  action.frame_range[1] > frame_end:
      frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
      frame_start = action.frame_range[0]

    frame_end = np.max([60, frame_end])
    print("--------", bvh_path, frame_start, frame_end, )

    bpy.ops.export_anim.bvh(filepath=bvh_path,
                            frame_start=int(frame_start),
                            frame_end=int(frame_end), root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])
    print(data_path+"/"+file+" processed.")

if __name__ == '__main__':
    # data_path = "/Users/letiziagirardi/Downloads"
    # file = "misure_nuove2.fbx"
    # fbx2bvh(data_path, file)

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <data_path> <file>")
        sys.exit(1)

    # Extract the command-line arguments
    data_path = sys.argv[1]
    file = sys.argv[2]

    fbx2bvh(data_path, file)
