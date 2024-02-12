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
        plot_fig_2(dir_path, "kdevs")
    else:
        print("pass folder as arg")


def init_fig1(length):
    fig, ax = plt.subplots()
    ax.set_ylim( [0.375,0.625] )
    ax.set_xlabel( f'Bin index ({length:.0e} Bits/Bin)' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_fig1_fft():
    fig, ax = plt.subplots()
    #ax.set_xlim( [0,100] )
    ax.set_ylim( [0,2.5] )
    ax.set_xlabel( f'frequency' )
    return fig, ax

def init_fig2_Temps():
    fig, ax = plt.subplots()
    ax.set_ylim( [0.375,0.625] )
    ax.set_xlabel( f'Temperature [K]' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_fig2_kdevs():
    fig, ax = plt.subplots()
    ax.set_xlabel( f'Anisotropy Energy % Deviation' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_fig2_diff_Temps():
    fig, ax = plt.subplots()
    ax.set_ylim( [10**-5,10**0] )
    ax.set_xlabel( f'Temperature [K]' )
    ax.set_ylabel( 'Absolute difference from 0.5' )
    return fig, ax

def init_fig2_diff_kdevs():
    fig, ax = plt.subplots()
    ax.set_ylim( [10**-5,10**0] )
    ax.set_xlabel( f'Anisotropy Energy % Deviation' )
    ax.set_ylabel( 'Absolute difference from 0.5' )
    return fig, ax

def plot_fig_1(data_path):

    metadata = np.load(glob.glob(data_path + '/*metadata*.npz')[0])
    try:
        depth = int(metadata['depth'])
    except(ValueError):
        depth = 0
    length = metadata['length']
    method = str(metadata['method'])
    T = int(metadata['T'])
    kdev = int(metadata['kdev'])
    fig, ax = init_fig1(length)
    fig_fft, ax_fft = init_fig1_fft()

    probs = np.load(glob.glob(data_path + '/*streamdata*.npz')[0])
    ax.plot( probs["probs"], color='blue')

    ax.legend()
    plt.title(f"Bit stream w/ T={T}, kdev = {kdev}, {depth} XOR(s) {label_dict[method]}")

    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig1.svg")
    ax_fft.plot( fft( probs["probs"] ), color='blue')
    helper.prompt_save_svg(fig_fft, f"{data_path}/fig1_fft.svg")


def plot_fig_2(data_path, kind):

    metadatas = np.sort(glob.glob(data_path + '/*metadata*.npz'))
    x = np.load(metadatas[0])[kind]
    base = [0.5 for _ in range(len(x))]

    depths = []
    for m in metadatas:
        try:
            depths.append(np.load(m)['depth'])
        except(ValueError):
            depths.append(0)
    methods = [str((np.load(m))['method']) for m in metadatas]
    data_files = np.sort(glob.glob(data_path + '/*Sweep*.npz'))
    if kind == "Temps":
        fig, ax = init_fig2_Temps()
        fig_diff, ax_diff = init_fig2_diff_Temps()
        plt.title(f"Temperature Whitening")
        ys = [np.load(data)['probs_per_temp'] for data in data_files]
    elif kind == "kdevs":
        fig, ax = init_fig2_kdevs()
        fig_diff, ax_diff = init_fig2_diff_kdevs()
        plt.title(f"Device Variation Whitening")
        ys = [np.load(data)['probs_per_kdev'] for data in data_files]
    else:
        print("ERRORORORO")

    colors = ['black', 'blue', 'red']
    labels = [ label_dict[method] + ", " + str(depths[i]) + " XOR" for i,method in enumerate(methods) ]
    lines = [ ax.scatter(x, y, label=labels[i], color=colors[i], s=36, marker='^') for i,y in enumerate(ys)  ]

    ax.legend()
    ax.axhline(0.5, linestyle = 'dashed', alpha = 0.33)

    lines = [ ax_diff.scatter(x, abs(base-y), label=labels[i], color=colors[i], s=36, marker='^') for i,y in enumerate(ys)  ]
    ax_diff.set_yscale('log')

    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig2.svg")
    helper.prompt_save_svg(fig_diff, f"{data_path}/fig2_diff.svg")


if __name__ == "__main__":
    main()
