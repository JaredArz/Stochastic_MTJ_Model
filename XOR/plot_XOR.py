import sys
sys.path.append("../")
import glob
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import misc_funcs as helper
import numpy as np

word_size = 1

label_dict = { "2S1D" : "Two streams w/ 1 dev",
               "2S2D" : "Two streams w/ 2 dev",
               "OSS"  : "One stream split",
               "NO"   : "No XOR",
               "BNO"  : "Binned no XOR"}
def main():
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        plot_fig_X_from_parallel(dir_path)
        plot_fig_1_from_parallel(dir_path)
        plot_fig_2_T_from_parallel(dir_path)
        plot_fig_2_kdev_from_parallel(dir_path)
    else:
        print("pass folder as arg")


def init_fig1(length):
    fig, ax = plt.subplots()
    ax.set_ylim( [0.3,0.7] )
    ax.set_xlabel( f'Bin index ({length:.0e} Bits/Bin)' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_fig1_fft():
    fig, ax = plt.subplots()
    #ax.set_xlim( [0,100] )
    ax.set_ylim( [0,8] )
    ax.set_xlabel( f'frequency' )
    return fig, ax

def plot_fig_1(data_path):

    metadata = np.load(glob.glob(data_path + '/*metadata*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth = int(metadata['depth'])
    except(ValueError):
        depth = 0

    length  = metadata['length']
    method  = str(metadata['method'])
    T       = int(metadata['T'])
    kdev    = int(metadata['kdev'])
    fig, ax = init_fig1(length)
    fig_fft, ax_fft = init_fig1_fft()

    probs = np.load(glob.glob(data_path + '/*streamdata*.npz')[0])
    ax.plot( probs["probs"], color='blue')

    ax.legend()
    plt.title(f"Bit stream w/ T={T}, kdev = {kdev}, {depth} XOR(s) {label_dict[method]}")

    ax_fft.plot( fft( probs["probs"] ), color='blue')
    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig1.svg")
    helper.prompt_save_svg(fig_fft, f"{data_path}/fig1_fft.svg")

def plot_fig_1_from_parallel(data_path):

    m = 10
    mdata_path = './ws_res/ws_' + str(m)
    metadata = np.load(glob.glob(mdata_path + '/*metadata_single*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth = int(metadata['depth'])
    except(ValueError):
        depth = 0
    #depth = 0
    #depth2 = 0

    length  = metadata['length']
    method  = str(metadata['method'])
    T       = int(metadata['T'])
    kdev    = int(metadata['kdev'])
    fig, ax = init_fig1(length)
    fig_fft, ax_fft = init_fig1_fft()

    m2 = 20
    mdata_path2 = './ws_res/ws_' + str(m2)
    metadata2 = np.load(glob.glob(mdata_path2 + '/*metadata_single*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth2 = int(metadata2['depth'])
    except(ValueError):
        depth2 = 0

    length2  = metadata2['length']
    method2  = str(metadata2['method'])
    T2       = int(metadata2['T'])
    kdev2    = int(metadata2['kdev'])

    m3 = 0
    mdata_path3 = './ws_res/ws_' + str(m3)
    metadata3 = np.load(glob.glob(mdata_path3 + '/*metadata_single*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth3 = int(metadata3['depth'])
    except(ValueError):
        depth3 = 0

    length3  = metadata3['length']
    method3  = str(metadata3['method'])
    T3       = int(metadata3['T'])
    kdev3    = int(metadata3['kdev'])

    probs = np.load(glob.glob(mdata_path + '/*streamdata*.npz')[0])
    probs2 = np.load(glob.glob(mdata_path2 + '/*streamdata*.npz')[0])
    probs3 = np.load(glob.glob(mdata_path3 + '/*streamdata*.npz')[0])

    print('f1 dset 1: ')
    print(probs3["probs"])
    print('f1 dset 2: ')
    print(probs["probs"])
    print('f1 dset 3: ')
    print(probs2["probs"])

    ax.plot( probs3["probs"], color='blue', label = f"T={T3}")
    ax.plot( probs["probs"], color='black', label = f"T={T}")
    ax.plot( probs2["probs"], color='red', label = f"T={T2}")
    ax.legend()
    #plt.title(f"Bit stream w/ T={T}, kdev = {kdev}, {depth} XOR(s) {label_dict[method]}")

    ax_fft.plot( fft( probs["probs"] ), color='blue')
    #helper.prompt_show()
    helper.prompt_save_svg(fig, f"./ws_res/fig1.svg")
    helper.prompt_save_svg(fig_fft, f"./ws_res/fig1_fft.svg")

def plot_fig_X_from_parallel(data_path):

    m = 83
    mdata_path = './ws_res/ws_' + str(m)
    metadata = np.load(glob.glob(mdata_path + '/*metadata_single*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth = int(metadata['depth'])
    except(ValueError):
        depth = 0

    length  = metadata['length']
    method  = str(metadata['method'])
    T       = int(metadata['T'])
    kdev    = int(metadata['kdev'])
    fig, ax = init_fig1(length)

    m2 = 85
    mdata_path2 = './ws_res/ws_' + str(m2)
    metadata2 = np.load(glob.glob(mdata_path2 + '/*metadata_single*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth2 = int(metadata2['depth'])
    except(ValueError):
        depth2 = 0

    length2  = metadata2['length']
    method2  = str(metadata2['method'])
    T2       = int(metadata2['T'])
    kdev2    = int(metadata2['kdev'])

    m3 = 80
    mdata_path3 = './ws_res/ws_' + str(m3)
    metadata3 = np.load(glob.glob(mdata_path3 + '/*metadata_single*.npz')[0])
    # hack because depth is stored as None if no xor and code crashes if try to load None
    try:
        depth3 = int(metadata3['depth'])
    except(ValueError):
        depth3 = 0

    length3  = metadata3['length']
    method3  = str(metadata3['method'])
    T3       = int(metadata3['T'])
    kdev3    = int(metadata3['kdev'])

    probs = np.load(glob.glob(mdata_path + '/*streamdata*.npz')[0])
    probs2 = np.load(glob.glob(mdata_path2 + '/*streamdata*.npz')[0])
    probs3 = np.load(glob.glob(mdata_path3 + '/*streamdata*.npz')[0])

    ax.plot( probs3["probs"], color='green', label = f"T={T3}")
    ax.plot( probs["probs"], color='black', label = f"T={T}")
    ax.plot( probs2["probs"], color='purple', label = f"T={T2}")
    ax.legend()
    helper.prompt_save_svg(fig, f"./ws_res/figX.svg")


def plot_fig_2_kdev(data_path):
    metadata   = np.load(data_path + '/metadata_kdev_sweep.npz')
    data_files = np.sort(glob.glob(data_path + '/*plottable*.npz'))
    data = [ np.load(f) for f in data_files ]

    fig, ax = plt.subplots()
    ax.set_xlabel( f'Anisotropy Energy % Deviation' )
    ax.set_ylabel( 'p' )
    fig_diff, ax_diff = plt.subplots()
    ax_diff.set_ylim( [10**-5,10**0] )
    ax_diff.set_xlabel( f'Anisotropy Energy % Deviation' )
    ax_diff.set_ylabel( 'Absolute difference from 0.5' )
    plt.title(f"Device Variation Whitening")
    x = metadata["kdevs"]

    base = [0.5 for _ in range(len(x))]
    colors = ['black', 'blue', 'red']
    # function to take data point and return a formatted string to use as label 
    format_label = lambda D: label_dict[ str(D["method"]) ] + ", " + str(D["depth"]) + " XOR"
    for i,D in enumerate(data) : ax.scatter(x, D["probs_per_kdev"], label=format_label(D), color=colors[i], s=36, marker='^')
    for i,D in enumerate(data) : ax_diff.scatter(x, abs(base-(D["probs_per_kdev"])), label=format_label(D), color=colors[i], s=36, marker='^')

    ax.legend()
    ax.axhline(0.5, linestyle = 'dashed', alpha = 0.33)
    ax_diff.set_yscale('log')
    ax_diff.legend()

    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig2.svg")
    helper.prompt_save_svg(fig_diff, f"{data_path}/fig2_diff.svg")

def plot_fig_2_T(data_path):
    metadata = np.load(data_path + '/metadata_sweep.npz')
    data_files = np.sort(glob.glob(data_path + '/*plottable*.npz'))
    data = [ np.load(f) for f in data_files ]

    fig, ax = plt.subplots()
    ax.set_ylim( [0.375,0.625] )
    ax.set_xlabel( f'Temperature [K]' )
    ax.set_ylabel( 'p' )
    fig_diff, ax_diff = plt.subplots()
    ax_diff.set_ylim( [10**-5,10**0] )
    ax_diff.set_xlabel( f'Temperature [K]' )
    ax_diff.set_ylabel( 'Absolute difference from 0.5' )
    plt.title(f"Temperature Whitening")
    x = metadata["Temps"]

    base = [0.5 for _ in range(len(x))]
    colors = ['black', 'blue', 'red']
    format_label = lambda D: label_dict[ str(D["method"]) ] + ", " + str(D["depth"]) + " XOR"
    for i,D in enumerate(data) : ax.scatter(x, D["probs_per_temp"], label=format_label(D), color=colors[i], s=36, marker='^')
    for i,D in enumerate(data) : ax_diff.scatter(x, abs(base-(D["props_per_temp"])), label=format_label(D), color=colors[i], s=36, marker='^')

    ax.legend()
    ax.axhline(0.5, linestyle = 'dashed', alpha = 0.33)
    ax_diff.set_yscale('log')
    ax_diff.legend()

    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig2.svg")
    helper.prompt_save_svg(fig_diff, f"{data_path}/fig2_diff.svg")

def plot_fig_2_T_from_parallel(data_path):
    
    fig, ax = plt.subplots()
    ax.set_ylim( [0.45,0.55] )
    ax.set_xlim( [288, 312])
    ax.set_xlabel( f'Temperature [K]' )
    ax.set_ylabel( 'p' )
    plt.xticks([290, 295, 300, 305, 310])
    fig_diff, ax_diff = plt.subplots()
    ax_diff.set_ylim( [10**-5,10**0] )
    ax_diff.set_xlim( [288, 312])
    ax_diff.set_xlabel( f'Temperature [K]' )
    ax_diff.set_ylabel( 'Absolute difference from 0.5' )
    plt.xticks([290, 295, 300, 305, 310])
    #plt.title(f"Temperature Whitening")

    
    for m in range(25):
        cind = m%5
        mdata_path = './ws_res/ws_' + str(m)
        metadata = np.load(mdata_path + '/metadata_sweep.npz')
        data_files = np.sort(glob.glob(mdata_path + '/*plottable_T*.npz'))
        data = [ np.load(f) for f in data_files ]

        x = metadata["Temps"]

        base = 0.5
        colors = ['black', 'black', 'black', 'red', 'red']
        markers = ['o', '^', 's', '^', 's']
        format_label = lambda D: label_dict[ str(D["method"]) ] + ", " + str(D["depth"]) + " XOR"
        for i,D in enumerate(data) :
            if (cind != 3):
              ax.errorbar(x, D["probs_per_temp"], yerr = D["std_per_temp"], fmt = markers[cind], capsize = 3, ecolor = colors[cind], label='_'+format_label(D), color=colors[cind], ms=5)
              #print('dpoint abs: ')
              #print(x, D["probs_per_temp"],D["std_per_temp"],cind)
        for i,D in enumerate(data) :
            if (cind != 3):
              ax_diff.scatter(x, abs(base-(D["probs_per_temp"])), label='_'+format_label(D), color=colors[cind], s=36, marker=markers[cind])


    fig_leg, ax_leg = plt.subplots()

    ax_leg.scatter(0, 0, label='No XOR, 0 XOR',color=colors[0], s=36, marker = markers[0])
    #ax_leg.scatter(0, 0, label='Two streams w/ 1 dev, 1 XOR',color=colors[1], s=36, marker = markers[1])
    ax_leg.scatter(0, 0, label='Two streams w/ 1 dev, 2 XOR',color=colors[2], s=36, marker = markers[2])
    ax_leg.scatter(0, 0, label='Two streams w/ 2 dev, 1 XOR',color=colors[3], s=36, marker = markers[3])
    ax_leg.scatter(0, 0, label='Two streams w/ 2 dev, 2 XOR',color=colors[4], s=36, marker = markers[4])
    ax_leg.legend()
    ax.axhline(0.5, linestyle = 'dashed', alpha = 0.33)
    ax_diff.scatter(1, 1, label='No XOR, 0 XOR',color=colors[0], s=36, marker = markers[0])
    ax_diff.scatter(1, 1, label='Two streams w/ 1 dev, 1 XOR',color=colors[1], s=36, marker = markers[1])
    ax_diff.scatter(1, 1, label='Two streams w/ 1 dev, 2 XOR',color=colors[2], s=36, marker = markers[2])
    ax_diff.scatter(1, 1, label='Two streams w/ 2 dev, 1 XOR',color=colors[3], s=36, marker = markers[3])
    ax_diff.scatter(1, 1, label='Two streams w/ 2 dev, 2 XOR',color=colors[4], s=36, marker = markers[4])
    ax_diff.set_yscale('log')
    #ax_diff.legend()

    #helper.prompt_show()
    helper.prompt_save_svg(fig, "./ws_res/fig2a.svg")
    helper.prompt_save_svg(fig_diff, "./ws_res/fig2a_diff.svg")
    helper.prompt_save_svg(fig_leg, "./ws_res/fig2a_leg.svg")

def plot_fig_2_kdev_from_parallel(data_path):

    fig, ax = plt.subplots()
    ax.set_ylim( [0.3,0.7] )
    ax.set_xlim( [-0.055, 0.055])
    ax.set_xlabel( f'Anisotropy Energy % Deviation' )
    ax.set_ylabel( 'p' )
    plt.xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    fig_diff, ax_diff = plt.subplots()
    ax_diff.set_ylim( [10**-5,10**0] )
    ax_diff.set_xlim( [-0.055, 0.055])
    ax_diff.set_xlabel( f'Anisotropy Energy % Deviation' )
    ax_diff.set_ylabel( 'Absolute difference from 0.5' )
    plt.xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    #plt.title(f"Device Variation Whitening")


    for m in range(25,80):
        cind = m%5
        mdata_path = './ws_res/ws_' + str(m)
        metadata = np.load(mdata_path + '/metadata_sweep.npz')
        data_files = np.sort(glob.glob(mdata_path + '/*plottable_kdev*.npz'))
        data = [ np.load(f) for f in data_files ]

        x = metadata["kdevs"]

        base = 0.5
        colors = ['black', 'black', 'black', 'red', 'red']
        markers = ['o', '^', 's', '^', 's']
        format_label = lambda D: label_dict[ str(D["method"]) ] + ", " + str(D["depth"]) + " XOR"
        for i,D in enumerate(data) :
            if (cind != 3) :
              ax.errorbar(x, D["probs_per_kdev"], yerr = D["std_per_kdev"], fmt = markers[cind], capsize = 3, ecolor = colors[cind], label='_'+format_label(D), color=colors[cind], ms=5)
              #print('dpoint abs: ')
              #print(x, D["probs_per_kdev"],D["std_per_kdev"],cind)
        for i,D in enumerate(data) :
            if (cind != 3) :
              ax_diff.scatter(x, abs(base-(D["probs_per_kdev"])), label='_'+format_label(D), color=colors[cind], s=36, marker=markers[cind])

    for m in range(80,91):
        cind = 0
        mdata_path = './ws_res/ws_' + str(m)
        metadata = np.load(mdata_path + '/metadata_sweep.npz')
        data_files = np.sort(glob.glob(mdata_path + '/*plottable_kdev*.npz'))
        data = [ np.load(f) for f in data_files ]

        x = metadata["kdevs"]

        base = 0.5
        format_label = lambda D: label_dict[ str(D["method"]) ] + ", " + str(D["depth"]) + " XOR"
        for i,D in enumerate(data) : ax.errorbar(x, D["probs_per_kdev"], yerr = D["std_per_kdev"], fmt = 'o', capsize = 3, ecolor = 'blue', label='_'+format_label(D), color='blue', ms=5)
        for i,D in enumerate(data) : ax_diff.scatter(x, abs(base-(D["probs_per_kdev"])), label='_'+format_label(D), color='blue', s=36, marker='o')
        #print('dpoint abs: ')
        #print(x, D["probs_per_kdev"],D["std_per_kdev"],cind)


    fig_leg, ax_leg = plt.subplots()
    ax_leg.scatter(0, 0, label='No XOR, 0 XOR',color=colors[0], s=36, marker = markers[0])
    ax_leg.scatter(0, 0, label='Two streams w/ 1 dev, 1 XOR',color=colors[1], s=36, marker = markers[1])
    ax_leg.scatter(0, 0, label='Two streams w/ 1 dev, 2 XOR',color=colors[2], s=36, marker = markers[2])
    #ax_leg.scatter(0, 0, label='Two streams w/ 2 dev, 1 XOR',color=colors[3], s=36, marker = markers[3])
    ax_leg.scatter(0, 0, label='Two streams w/ 2 dev, 2 XOR',color=colors[4], s=36, marker = markers[4])
    ax_leg.scatter(0, 0, label='No XOR, binned',color='blue', s=36, marker = 'o')
    ax_leg.legend()
    ax.axhline(0.5, linestyle = 'dashed', alpha = 0.33)
    ax_diff.scatter(1, 1, label='No XOR, 0 XOR',color=colors[0], s=36, marker = markers[0])
    ax_diff.scatter(1, 1, label='Two streams w/ 1 dev, 1 XOR',color=colors[1], s=36, marker = markers[1])
    ax_diff.scatter(1, 1, label='Two streams w/ 1 dev, 2 XOR',color=colors[2], s=36, marker = markers[2])
    ax_diff.scatter(1, 1, label='Two streams w/ 2 dev, 1 XOR',color=colors[3], s=36, marker = markers[3])
    ax_diff.scatter(1, 1, label='Two streams w/ 2 dev, 2 XOR',color=colors[4], s=36, marker = markers[4])
    ax_diff.scatter(1, 1, label='No XOR, binned',color='blue', s=36, marker = 'o')
    ax_diff.set_yscale('log')
    #ax_diff.legend()

    #helper.prompt_show()
    helper.prompt_save_svg(fig, "./ws_res/fig2b.svg")
    helper.prompt_save_svg(fig_diff, "./ws_res/fig2b_diff.svg")
    helper.prompt_save_svg(fig_leg, "./ws_res/fig2b_leg.svg")

if __name__ == "__main__":
    main()
