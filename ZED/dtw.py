import numpy as np
from numpy import pi
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tslearn.generators import random_walks
from tslearn import metrics
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

np.random.seed(1)
n_ts, sz = 100, 34

def plot_time_series(ts, ax, color, color_code=None, alpha=1.):
    if color_code is not None:
        colors = [color_code] * len(ts)
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
    for i in range(len(ts) - 1):
        ax.plot(ts[i:i+2, 0], ts[i:i+2, 1], ts[i:i+2, 2],
                marker='o', c=colors[color], alpha=alpha)

dataset_1 = random_walks(n_ts=n_ts, sz=sz, d=3)
print(dataset_1.shape)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=pi)  # Rescale the time series
dataset_scaled_1 = scaler.fit_transform(dataset_1)

# DTW using a function as the metric argument
path_1, sim_1 = metrics.dtw_path(dataset_scaled_1[0], dataset_scaled_1[1])

print(dataset_1[0])
print(dataset_1[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_time_series(dataset_1[0], ax, 0)
plot_time_series(dataset_1[1], ax, 30)

# Plot the DTW path
for (i, j) in path_1:
    ax.plot([dataset_1[0, i, 0], dataset_1[1, j, 0]],
            [dataset_1[0, i, 1], dataset_1[1, j, 1]],
            [dataset_1[0, i, 2], dataset_1[1, j, 2]],
            color='r', alpha=.5)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("DTW with Optimal Alignment")

plt.tight_layout()
plt.show()
