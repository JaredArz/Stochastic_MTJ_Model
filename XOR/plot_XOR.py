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
               "NO"   : "No XOR"}
def main():
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        plot_fig_2_kdev(dir_path)
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
    metadata = np.load(data_path + '/metadata_kdev_sweep.npz')
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


if __name__ == "__main__":
    main()
