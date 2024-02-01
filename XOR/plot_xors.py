import sys
sys.path.append("../")
import helper_funcs as helper
import glob
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np
from scipy.stats import chi2

word_size   = 8
record_size = 1000
dof = 256
prob = [2**(-1*word_size) for _ in range(dof)]
colors = ['black', 'grey', 'green', 'red', 'blue']

class dset():
        def __init__(self, fname, plot_label, alpha):
            self.label = plot_label
            self.alpha = alpha
            self.fname = fname
            self.data  = np.load(fname)
            self.uniformity = funcs.get_uniformity(self.data, word_size, record_size)
            self.chisq = funcs.compute_chi_squared(self.uniformity, word_size, record_size)
            self.chisq_label = f"{self.chisq:.2f}"
            self.p_val = chi2.sf(self.chisq, 256)
            self.p_label = f"{self.p_val:.3f}"
            self.x = self.uniformity / record_size

def main():
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        plot_single(dir_path)
    else:
        print("pass folder as arg")

    #metadata = np.load(glob.glob(dir_path + "/*metadata_voltage*")[0])

def T_uniformity_plot():


def init_freq_plot():
    fig, ax = plt.subplots()
    ax.set_xlim( [0,255] )
    ax.set_ylim( [0.001,0.009] )
    ax.set_xlabel( 'Generated 8-bit Number' )
    ax.set_ylabel( 'p' )
    return fig, ax

d = dset("./wordstreams/ws_17:23:31/XORd_stream.npy", 'test', 0.5)
fig, ax = init_freq_plot()
ax.plot(d.x)
plt.title("Temperature whitening, 1 XOR")
helper.prompt_show()


if __name__ == "__main__":
    main()
