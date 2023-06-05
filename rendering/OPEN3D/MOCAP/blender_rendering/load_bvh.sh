#!/bin/bash

# Path to Blender executable
BLENDER_PATH="../../../../../../../../../../../Applications/Blender.app/Contents/MacOS/Blender"

# Run Blender with --P option and arg1
"$BLENDER_PATH" -P load_bvh.py
