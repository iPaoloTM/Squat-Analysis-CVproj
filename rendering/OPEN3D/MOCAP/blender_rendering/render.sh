#!/bin/bash

# Path to Blender executable
BLENDER_PATH="../../../../../../../../../../../Applications/Blender.app/Contents/MacOS/Blender"

# Run Blender with --P option and arg1
"$BLENDER_PATH" -P render.py -- --bvh_path misure_nuove2.bvh --save_path outputs/output.mp4 --render_engine cycles --render --frame_end 2000 --resX 1920 --resY 1080
