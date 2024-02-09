import sys
sys.path.append("../")
import glob
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import mtj_helper as helper
import numpy as np

word_size = 1

label_dict = { "2S1D" : "Two streams w/ 1 dev",
               "2S2D" : "Two streams w/ 2 dev",
               "OSS"  : "One stream split",
               "NO"   : "No XOR"}

def main():
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        plot_fig_1(dir_path)
    else:
        print("pass folder as arg")


def init_fig1(length):
    fig, ax = plt.subplots()
    ax.set_ylim( [0.375,0.625] )
    ax.set_xlabel( f'Bin index ({length:.0e} Bits/Bin)' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_fig2():
    fig, ax = plt.subplots()
    ax.set_ylim( [0.375,0.625] )
    ax.set_xlabel( f'Temperature [K]' )
    ax.set_ylabel( 'p' )
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
    fig, ax = init_fig1(length)

    probs = np.load(glob.glob(data_path + '/*streamdata*.npz')[0])
    ax.plot( probs["probs"], color='blue')

    ax.legend()
    print(T)
    print(depth)
    print(method)
    plt.title(f"Bit stream w/ T={T}, {depth} XOR(s) {label_dict[method]}")

    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig1.svg")


def plot_fig_2(data_path):

    metadatas = np.sort(glob.glob(data_path + '/*metadata*.npz'))
    Temps   = np.load(metadatas[0])['Temps']

    depths = []
    for m in metadatas:
        try:
            depths.append(np.load(m)['depth'])
        except(ValueError):
            depths.append(0)
    methods = [str((np.load(m))['method']) for m in metadatas]
    fig, ax = init_fig2()

    data_files = np.sort(glob.glob(data_path + '/*Sweep*.npz'))
    ys = [np.load(data)['probs_per_temp'] for data in data_files]
    colors = ['black', 'blue', 'red']
    labels = [ label_dict[method] + ", " + str(depths[i]) + " XOR" for i,method in enumerate(methods) ]
    lines = [ ax.scatter(Temps, y, label=labels[i], color=colors[i], s=36, marker='^') for i,y in enumerate(ys)  ]

    ax.legend()
    ax.axhline(0.5, linestyle = 'dashed', alpha = 0.33)
    plt.title(f"Temperature whitening")

    helper.prompt_show()
    helper.prompt_save_svg(fig, f"{data_path}/fig2.svg")


if __name__ == "__main__":
    main()
