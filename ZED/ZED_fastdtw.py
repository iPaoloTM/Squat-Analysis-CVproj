import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import subprocess
import os
import json

def temporal_alignment(sequence1, sequence2):
    _, path = fastdtw(sequence1, sequence2, dist=euclidean)

    aligned_sequence1 = [sequence1[i] for i, _ in path]
    aligned_sequence2 = [sequence2[j] for _, j in path]

    return aligned_sequence1, aligned_sequence2

# Generate sinusoidal sequences
t = np.linspace(0, 2*np.pi, 100)
t1 = np.linspace(0, 2*np.pi, 50)
sequence1 = np.sin(t[:, None] + np.arange(34))
sequence2 = np.sin(t1[::-1, None] + np.arange(34))

skeleton1=[]
skeleton2=[]

# Perform temporal alignment
aligned_sequence1, aligned_sequence2 = temporal_alignment(sequence1, sequence2)

# Create a directory to store the frames
os.makedirs("frames", exist_ok=True)

# Save the frames of the original sequences
for i, (frame1, frame2) in enumerate(zip(sequence1, sequence2)):
    plt.figure()
    plt.plot(frame1, 'b-', label='Sequence 1')
    plt.plot(frame2, 'r-', label='Sequence 2')
    plt.xlabel("Joint Index")
    plt.ylabel("Joint Value")
    plt.title(f"Frame {i+1}")
    plt.legend()
    plt.savefig(f"frames/frame_{i}.png")
    plt.close()

# Save the frames of the aligned sequences
for i, (frame1, frame2) in enumerate(zip(aligned_sequence1, aligned_sequence2)):
    plt.figure()
    plt.plot(frame1, 'b-', label='Aligned Sequence 1')
    plt.plot(frame2, 'r-', label='Aligned Sequence 2')
    plt.xlabel("Joint Index")
    plt.ylabel("Joint Value")
    plt.title(f"Aligned Frame {i+1}")
    plt.legend()
    plt.savefig(f"frames/aligned_frame_{i}.png")
    plt.close()

# Use ffmpeg to create a video of the original sequences
subprocess.run([
    "ffmpeg",
    "-framerate", "5",
    "-i", "frames/frame_%d.png",
    "-c:v", "libx264",
    "-r", "30",
    "-pix_fmt", "yuv420p",
    "original_sequences.mp4"
])

# Use ffmpeg to create a video of the aligned sequences
subprocess.run([
    "ffmpeg",
    "-framerate", "5",
    "-i", "frames/aligned_frame_%d.png",
    "-c:v", "libx264",
    "-r", "30",
    "-pix_fmt", "yuv420p",
    "aligned_sequences.mp4"
])
