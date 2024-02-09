import sys
sys.path.append("../")
import helper_funcs as helper
import glob
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np
from scipy.stats import chi2

word_size = 8

label_dict = { "2S1D" : "Two streams w/ one dev",
               "2S2D" : "Two streams w/ two dev",
               "OSS"  : "One stream split"}

def main():
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        T_uniformity_plot(dir_path)
    else:
        print("pass folder as arg")

def load_baseline():
    f_data = np.load('base_uniformity.npz')
    return f_data

def init_uniformity_plot():
    fig, ax = plt.subplots()
    ax.set_xlim( [0,255] )
    ax.set_ylim( [0.001,0.009] )
    ax.set_xlabel( 'Generated 8-bit Number' )
    ax.set_ylabel( 'p' )
    return fig, ax

def init_p_measure_plot():
    fig, ax = plt.subplots()
    ax.set_xlabel( 'T [K]' )
    ax.set_ylabel( 'p' )
    return fig, ax

def T_uniformity_plot(data_path):
    fig, ax = init_uniformity_plot()

    base_data = load_baseline()
    metadata = np.load(data_path + '/metadata.npz')
    depth = metadata['depth']
    method = metadata['method']
    T = metadata['T']

    xor = np.load(glob.glob(data_path + '/*XOR.npz')[0])
    L1  = np.load(glob.glob(data_path + '/*L1.npz')[0])
    R1  = np.load(glob.glob(data_path + '/*R1.npz')[0])
    data_set = [base_data, xor, L1, R1]
    opacity = [0.9, 0.7, 0.4, 0.4]
    colors = ['black', 'grey', 'red', 'blue']
    labels = ['base', 'xor', 'input a', 'input b']

    if depth == 2:
        pass
        ''' FIXME
        L2 = np.load(glob.glob(data_path + '/*L2.npz')[0])
        R2 = np.load(glob.glob(data_path + '/*R2.npz')[0])
        data_set.append(L2)
        data_set.append(R2)
        '''

    lines = [ ax.plot(data['x'], alpha=opacity[i], label=labels[i], color=colors[i]) for i,data in enumerate(data_set)  ]
    stats = [ f"{data['chisq']:.2f}" + ' | ' + f"{data['p_val']:.2f}" for data in data_set ]
    legend1 = ax.legend()
    plt.legend([l[0] for l in lines], stats, loc=4)
    plt.gca().add_artist(legend1)
    #plt.title(f"Temperature whitening (T={T}), {depth} XOR(s) {label_dict[method]}")

    helper.prompt_show()
    helper.prompt_save_svg(fig,'test.svg')

def K_uniformity_plot(data_path):
    fig, ax = init_uniformity_plot()

    base_data = load_baseline()
    metadata = np.load(data_path + '/metadata.npz')
    depth = metadata['depth']
    method = metadata['method']
    K = metadata['Kdev']

    xor = np.load(glob.glob(data_path + '/*XOR.npz')[0])
    L1  = np.load(glob.glob(data_path + '/*L1.npz')[0])
    R1  = np.load(glob.glob(data_path + '/*R1.npz')[0])
    data_set = [base_data, xor, L1, R1]
    opacity = [0.9, 0.7, 0.4, 0.4]
    colors = ['black', 'grey', 'red', 'blue']
    labels = ['base', 'xor', 'input a', 'input b']

    if depth == 2:
        pass
        ''' FIXME
        L2 = np.load(glob.glob(data_path + '/*L2.npz')[0])
        R2 = np.load(glob.glob(data_path + '/*R2.npz')[0])
        data_set.append(L2)
        data_set.append(R2)
        '''

    lines = [ ax.plot(data['x'], alpha=opacity[i], label=labels[i], color=colors[i]) for i,data in enumerate(data_set)  ]
    stats = [ f"{data['chisq']:.2f}" + ' | ' + f"{data['p_val']:.2f}" for data in data_set ]
    legend1 = ax.legend()
    plt.legend([l[0] for l in lines], stats, loc=4)
    plt.gca().add_artist(legend1)
    #plt.title(f"Temperature whitening (T={T}), {depth} XOR(s) {label_dict[method]}")

    helper.prompt_show()
    helper.prompt_save_svg(fig,'test.svg')


def T_p_measure_plot(data_path):
    fig, ax = init_p_measure_plot()

    metadata = np.load(data_path + '/metadata.npz')
    depth  = metadata['depth']
    method = metadata['method']
    Temps = metadata['Temps']
    #Temps = [300, 315, 330]

    data = np.load(glob.glob(data_path + '/plottable*')[0])
    ps = data['x']

    ax.plot(Temps,ps)

    plt.title(f"meta p")

    helper.prompt_show()
    helper.prompt_save_svg(fig,'metaptest.svg')


def K_p_measure_plot(data_path):
    fig, ax = init_p_measure_plot()

    metadata = np.load(data_path + '/metadata.npz')
    depth  = metadata['depth']
    method = metadata['method']
    Kdevs = metadata['Kdevs']
    #Temps = [300, 315, 330]

    data = np.load(glob.glob(data_path + '/plottable*')[0])
    ps = data['x']

    ax.plot(Kdevs,ps)

    plt.title(f"meta p")

    helper.prompt_show()
    helper.prompt_save_svg(fig,'metaptest.svg')


if __name__ == "__main__":
    main()
