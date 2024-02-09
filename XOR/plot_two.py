import sys
sys.path.append("../")
import helper_funcs as helper
import glob
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np

word_size = 1

label_dict = { "2S1D" : "Two streams w/ one dev",
               "2S2D" : "Two streams w/ two dev",
               "OSS"  : "One stream split",
               "NO"   : "No XOR"}

def main():
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        plot_fig_2(dir_path)
    else:
        print("pass folder as arg")

'''
def load_baseline():
    f_data = np.load('base_uniformity.npz')
    return f_data
'''

def init_fig1(length):
    fig, ax = plt.subplots()
    ax.set_ylim( [0,1] )
    ax.set_xlabel( f'Bin index ({length:.0e} Bits/Bin)' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_fig2():
    fig, ax = plt.subplots()
    ax.set_ylim( [0,1] )
    ax.set_xlabel( f'Temperature [K]' )
    ax.set_ylabel( 'p' )
    return fig, ax

def plot_fig_1(data_path):

    #base_data = load_baseline()
    metadata = np.load(data_path + '/metadata.npz')
    #depth = metadata['depth']
    length = metadata['length']
    method = metadata['method']
    T = metadata['T']
    fig, ax = init_fig1(length)

    probs = np.load(glob.glob(data_path + '/*streamdata.npz')[0])
    ax.plot( probs["probs"], color='blue')

    ax.legend()
    #plt.title(f"Temperature whitening (T={T}), {depth} XOR(s) {label_dict[method]}")

    helper.prompt_show()
    helper.prompt_save_svg(fig,'test.svg')

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
    labels = [ label_dict[method] + " " + str(depths[i]) + "XOR" for i,method in enumerate(methods) ]
    lines = [ ax.scatter(Temps, y, label=labels[i], color=colors[i]) for i,y in enumerate(ys)  ]

    ax.legend()
    plt.title(f"Temperature whitening")

    helper.prompt_show()
    helper.prompt_save_svg(fig,'test.svg')


if __name__ == "__main__":
    main()
