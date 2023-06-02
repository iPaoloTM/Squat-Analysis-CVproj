import numpy as np
from numpy import pi
import json
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tslearn.generators import random_walks
from tslearn import metrics
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

np.random.seed(1)
n_ts, sz = 1000, 34

def read_skeletal_trajectories(file1, file2):

    # load the JSON file
    with open('../body_data/second_attempt/'+file1+'ZED.json', 'r') as f:
        data = json.load(f)

    # Extract the "keypoint" vectors
    keypoints = []
    for body in data.values():
        for body_part in body['body_list']:
            keypoints.append(body_part['keypoint'])
    keypoints1 = np.array(keypoints)

    # load the JSON file
    with open('../body_data/second_attempt/'+file2+'ZED.json', 'r') as f:
        data = json.load(f)

    # Extract the "keypoint" vectors
    keypoints = []
    for body in data.values():
        for body_part in body['body_list']:
            keypoints.append(body_part['keypoint'])
    keypoints2 = np.array(keypoints)

    # print(keypoints1.shape)
    # print(keypoints2.shape)

    return keypoints1, keypoints2

def plot_time_series(ts, ax, color, color_code=None, alpha=1.):
    if color_code is not None:
        colors = [color_code] * len(ts)
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
    for i in range(len(ts) - 1):
        ax.plot(ts[i:i+2, 0], ts[i:i+2, 1], ts[i:i+2, 2],
                marker='o', c=colors[color], alpha=alpha)

def main():

    # Assuming you have two time series of shape (1000, 34, 3)
    time_series_1 = np.random.rand(100, 34, 3)
    time_series_2 = np.random.rand(100, 34, 3)

    # Reshape the time series to have shape (n_samples * n_timestamps, n_features)
    reshaped_ts_1 = time_series_1.reshape(-1, 3)
    reshaped_ts_2 = time_series_2.reshape(-1, 3)

    # Compute the DTW path
    path, dist = metrics.dtw_path(reshaped_ts_1, reshaped_ts_2)

    # Reshape the path to have shape (n_samples, n_timestamps, 2)
    n_samples = time_series_1.shape[0]
    n_timestamps = time_series_1.shape[1]

    # Plot the time series and the optimal alignment for the first sample
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot time series 1
    for i in range(n_timestamps - 1):
        ax.plot(time_series_1[0, i:i+2, 0], time_series_1[0, i:i+2, 1], time_series_1[0, i:i+2, 2],
                marker='o', c='b', alpha=0.5)

    # Plot time series 2
    for i in range(n_timestamps - 1):
        ax.plot(time_series_2[0, i:i+2, 0], time_series_2[0, i:i+2, 1], time_series_2[0, i:i+2, 2],
                marker='o', c='r', alpha=0.5)

    print(np.array(path[6]).shape)
    exit(0)
    # Plot the optimal alignment
    for (i, j) in path[0]:
        ax.plot([time_series_1[pair_index, i, 0], time_series_2[pair_index, j, 0]],
                [time_series_1[pair_index, i, 1], time_series_2[pair_index, j, 1]],
                [time_series_1[pair_index, i, 2], time_series_2[pair_index, j, 2]],
                color='g', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('DTW with Optimal Alignment')
    plt.show()



    # read_skeletal_trajectories("groundTruth","sample")
    #
    # dataset_1 = random_walks(n_ts=100, sz=34, d=3)
    # dataset_2 = random_walks(n_ts=94, sz=34, d=3)
    # print(dataset_1.shape)
    # print(dataset_2.shape)
    #
    # # DTW using a function as the metric argument
    # path_1, sim_1 = metrics.dtw_path(dataset_1, dataset_2)
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # plot_time_series(dataset_1, ax, 0)
    # plot_time_series(dataset_2, ax, 30)
    #
    # # Plot the DTW path
    # for (i, j) in path_1:
    #     ax.plot([dataset_1[0, i, 0], dataset_1[1, j, 0]],
    #             [dataset_1[0, i, 1], dataset_1[1, j, 1]],
    #             [dataset_1[0, i, 2], dataset_1[1, j, 2]],
    #             color='r', alpha=.5)
    #
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title("DTW with Optimal Alignment")
    #
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
