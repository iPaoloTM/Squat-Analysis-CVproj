import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

def main():

    reference_keypositions=[[502, 1527], [781, 3445], [798, 3564], [826, 3872], [865, 4041], [880, 4174], [920, 4536], [1134, 6119], [1164, 6290], [1201, 6716], [1247, 6887], [1262, 7020], [1305, 7614], [1512, 8921], [1535, 9062], [1587, 9430], [1615, 9647], [1633, 9809], [1678, 10977]]

    sample_keypositions=[[527, 0], [847, 4250], [872, 4424], [901, 4645], [971, 5174], [992, 5325], [1034, 5914], [1208, 6919], [1226, 7031], [1250, 7350], [1293, 7531], [1302, 7610], [1313, 7931], [1492, 9207], [1549, 9549], [1604, 9983], [1653, 10208], [1665, 10337], [1696, 11142]]

    ZED_total=np.array([
    0.05315317763239643,
    0.01743043089091098,
    0.025249631396809163,
    0.045532318744174044,
    0.044897681755733665,
    0.030451246577178584,
    0.009933511204974887,
    0.018084108584631036,
    0.019924256804912988,
    0.0385150366789859,
    0.027573824241141867,
    0.015852340123424916,
    0.017688414993921292,
    0.012401369694062234,
    0.024257968407205927,
    0.03697979936180227,
    0.03175820460596027,
    0.02821526324270443,
    0.024286920405079515,
    ])

    MOCAP_total=np.array([
    0.06898930078765877,
    0.01784328716576746,
    0.026423708621471465,
    0.024526475722053877,
    0.02118775497216663,
    0.017339834880434446,
    0.009421447658987966,
    0.023218997204270918,
    0.03886382210294408,
    0.05755210136204917,
    0.036261451760614155,
    0.020850474971375432,
    0.007888049206864898,
    0.014402736700958733,
    0.023136325170851225,
    0.03745456584864545,
    0.019528716191841405,
    0.013887933387082265,
    0.05042553091214627,
    ])

    ZED_lower=np.array([
    0.007778223723627992,
    0.03610720897597028,
    0.02923803960868991,
    0.04230557528976832,
    0.07679846878556336,
    0.03743178417478878,
    0.01733375134235179,
    0.05161581145500551,
    0.06380261074191594,
    0.10681658021706818,
    0.0948123689110231,
    0.04288315993218684,
    0.026295332666483155,
    0.032571384313028626,
    0.0612029385910394,
    0.09569388032134206,
    0.06989460075233374,
    0.04353300975569802,
    0.02836854459313988,
    ])

    MOCAP_lower=np.array([
    0.009528691899836518,
    0.02829832148944826,
    0.038609196504774866,
    0.05318374230730553,
    0.0353468945778355,
    0.026727500029100737,
    0.014326069725565237,
    0.04296915454073405,
    0.07510456409364973,
    0.10799744442774105,
    0.07398129258418915,
    0.04653537891653182,
    0.015533905289661766,
    0.03538422764294533,
    0.05846688944450332,
    0.07229517092599329,
    0.04826150228672627,
    0.03298213875358439,
    0.047973758263513945,
    ])

    fig = plt.figure(figsize=(12, 6))
    time = np.arange(len(reference_keypositions))

    # ax1 = fig.add_subplot(121)
    # ax1.plot(time, ZED_total, color='orange')
    # ax1.plot(time, MOCAP_total, color='green')
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('MPJPE')
    # plt.title("MPJPE Total body")
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(time, ZED_lower, color='orange')
    # ax2.plot(time, MOCAP_lower, color='green')
    # ax2.set_xlabel('T-pose')
    # ax2.set_ylabel('MPJPE')
    # plt.title("MPJPE Lower body")

    ax3 = fig.add_subplot(111)
    ax3.plot(time, ZED_total-MOCAP_total, color='red')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('MPJPE_ZED - MPJPE_MOCAP')
    plt.title("MPJPE Difference")

    # ax3 = fig.add_subplot(122)
    # ax3.plot(time, ZED_lower-MOCAP_lower, color='red')
    # ax3.set_xlabel('Time')
    # ax3.set_ylabel('MPJPE_ZED - MPJPE_MOCAP')

    plt.show()

if __name__ == '__main__':
    main()
