import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# define the points
points = [
[
                        679.83447265625,
                        452.6777038574219
                    ],
                    [
                        681.1698608398438,
                        431.9984130859375
                    ],
                    [
                        682.452880859375,
                        411.0675964355469
                    ],
                    [
                        683.7525024414062,
                        389.8669738769531
                    ],
                    [
                        688.7412109375,
                        389.3916015625
                    ],
                    [
                        698.8731689453125,
                        388.6606140136719
                    ],
                    [
                        712.48876953125,
                        396.8653869628906
                    ],
                    [
                        728.0387573242188,
                        398.27606201171875
                    ],
                    [
                        731.218994140625,
                        398.5645751953125
                    ],
                    [
                        737.6519165039062,
                        399.1481628417969
                    ],
                    [
                        733.3233032226562,
                        402.59796142578125
                    ],
                    [
                        678.72265625,
                        390.1144104003906
                    ],
                    [
                        668.1251220703125,
                        390.8789978027344
                    ],
                    [
                        673.7620849609375,
                        387.95220947265625
                    ],
                    [
                        668.3170166015625,
                        393.6191711425781
                    ],
                    [
                        667.1878662109375,
                        394.7943420410156
                    ],
                    [
                        664.887451171875,
                        397.1884765625
                    ],
                    [
                        670.188720703125,
                        396.40423583984375
                    ],
                    [
                        689.6520385742188,
                        451.49945068359375
                    ],
                    [
                        705.6615600585938,
                        505.76678466796875
                    ],
                    [
                        711.0620727539062,
                        553.0303344726562
                    ],
                    [
                        721.156494140625,
                        570.1730346679688
                    ],
                    [
                        669.8193359375,
                        453.8796691894531
                    ],
                    [
                        661.946533203125,
                        509.2998352050781
                    ],
                    [
                        653.8782348632812,
                        557.98974609375
                    ],
                    [
                        658.8139038085938,
                        576.7842407226562
                    ],
                    [
                        688.0179443359375,
                        382.4822692871094
                    ],
                    [
                        688.6417236328125,
                        379.5346374511719
                    ],
                    [
                        689.3502197265625,
                        376.8482666015625
                    ],
                    [
                        688.4929809570312,
                        376.4933166503906
                    ],
                    [
                        686.3576049804688,
                        376.7510681152344
                    ],
                    [
                        680.0574951171875,
                        376.2157897949219
                    ],
                    [
                        709.5728759765625,
                        562.9649047851562
                    ],
                    [
                        649.622802734375,
                        568.0103759765625
                    ]
]

# split the points into x and y coordinates
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]

# create a new figure
fig, ax = plt.subplots()

# plot the points as red circles

# create a colormap with 18 colors
colormap = cm.get_cmap('jet', 18)

# normalize data to colormap range
norm = Normalize(vmin=0, vmax=34)

# iterate over data and plot each point
for i, point in enumerate(points):
    ax.scatter(point[0], point[1], c=colormap(norm(i)), marker='o')

# add a colorbar with a label
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap))
cbar.ax.set_ylabel('Point number')

# set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

# ax.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], 'b-') #right shoulder
# ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], 'b-') #right arm
# ax.plot([points[3][0], points[4][0]], [points[3][1], points[4][1]], 'b-') #right forearm
# ax.plot([points[1][0], points[5][0]], [points[1][1], points[5][1]], 'b-') #left shoulder
# ax.plot([points[5][0], points[6][0]], [points[5][1], points[6][1]], 'b-') #left arm
# ax.plot([points[6][0], points[7][0]], [points[6][1], points[7][1]], 'b-') #left forearm
# ax.plot([points[1][0], points[8][0]], [points[1][1], points[8][1]], 'b-') #right part of back
# ax.plot([points[8][0], points[9][0]], [points[8][1], points[9][1]], 'b-') #right leg
# ax.plot([points[9][0], points[10][0]], [points[9][1], points[10][1]], 'b-') #right calf
# ax.plot([points[1][0], points[11][0]], [points[1][1], points[11][1]], 'b-') #left part of back
# ax.plot([points[11][0], points[12][0]], [points[11][1], points[12][1]], 'b-') #left leg
# ax.plot([points[12][0], points[13][0]], [points[12][1], points[13][1]], 'b-') #left calf
# ax.plot([points[16][0], points[17][0]], [points[16][1], points[17][1]], 'b-') #eye line

# set the x and y limits of the plot
ax.set_xlim([400, 800])
ax.set_ylim([200, 700])

# for i in range(len(points)):
#     ax.text(points[i][0], points[i][1], str(i), fontsize=9)

ax.invert_yaxis()

# show the plot
plt.savefig("plot2d.png")
plt.show()
