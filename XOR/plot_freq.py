import sys
sys.path.append("../")
import helper_funcs as helper
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np
from scipy.stats import chi2

word_size   = 8
record_size = 100000
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


def get_subplot():
    fig, ax = plt.subplots()
    ax.set_xlim( [0,255] )
    ax.set_ylim( [0.001,0.009] )
    ax.set_xlabel( 'Generated 8-bit Number' )
    ax.set_ylabel( 'p' )
    return fig, ax


base       = dset('./base_freq.npy',                            "base", 0.9)
T_290_D1   = dset("./wordstreams/ws_17:07:16/no_xor.npy",       "290K, No XOR", 0.5)
T_310_D1   = dset("./wordstreams/ws_17:07:44/no_xor.npy",       "310K, No XOR", 0.5)
T_290_D1_X = dset("./wordstreams/ws_17:07:16/XORd_1_stream.npy","290K, One XOR", 0.7)
T_310_D1_X = dset("./wordstreams/ws_17:07:44/XORd_1_stream.npy","310K, One XOR", 0.7)

T_290_D2   = dset("./wordstreams/ws_17:08:24/no_xor.npy",       "290K, No XOR", 0.5)
T_310_D2   = dset("./wordstreams/ws_17:08:47/no_xor.npy",       "310K, No XOR", 0.5)
T_290_D2_X = dset("./wordstreams/ws_17:08:24/XORd_2_stream.npy","290K, Two XORs", 0.7)
T_310_D2_X = dset("./wordstreams/ws_17:08:47/XORd_2_stream.npy","310K, Two XORs", 0.7)

K_i1_D1    = dset("./wordstreams/ws_17:59:17/no_xor.npy",       "(a) K 2.5%, No XOR", 0.5)
K_i2_D1    = dset("./wordstreams/ws_17:59:26/no_xor.npy",       "(b) K 2.5%, No XOR", 0.5)
K_i1_D1_X  = dset("./wordstreams/ws_17:59:17/XORd_1_stream.npy","(a) K 2.5%, One XOR", 0.7)
K_i2_D1_X  = dset("./wordstreams/ws_17:59:26/XORd_1_stream.npy","(b) K 2.5%, One XOR", 0.7)

K_i1_D2    = dset("./wordstreams/ws_11:59:10/no_xor.npy",        "(a) K 2.5%, No XOR", 0.5)
K_i2_D2    = dset("./wordstreams/ws_11:59:17/no_xor.npy",        "(b) K 2.5%, No XOR", 0.5)
K_i1_D2_X  = dset("./wordstreams/ws_11:59:10/XORd_2_stream.npy", "(a) K 2.5%, Two XORs", 0.7)
K_i2_D2_X  = dset("./wordstreams/ws_11:59:17/XORd_2_stream.npy", "(b) K 2.5%, Two XORs", 0.7)


fig, ax = get_subplot()
data_set_one = [base, T_290_D1, T_310_D1, T_290_D1_X, T_310_D1_X]
lines = [ ax.plot(d.x, alpha=d.alpha, label=d.label, color = colors[i]) for i,d in enumerate(data_set_one)  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_one ]
legend1 = ax.legend()
plt.legend([l[0] for l in lines], stats, loc=4)
plt.title("Temperature whitening, 1 XOR")
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'./TD1.svg')

fig, ax = get_subplot()
data_set_two = [base, T_290_D2, T_310_D2, T_290_D2_X, T_310_D2_X]
lines = [ ax.plot(d.x, alpha=d.alpha, label=d.label, color=colors[i]) for i,d in enumerate(data_set_two)  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_two ]
legend1 = ax.legend()
plt.title("Temperature whitening, 2 XOR")
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'TD2.svg')

fig, ax = get_subplot()
data_set_three = [base, K_i1_D1, K_i2_D1, K_i1_D1_X, K_i2_D1_X]
lines = [ ax.plot(d.x, alpha=d.alpha, label=d.label, color=colors[i]) for i,d in enumerate(data_set_three)  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_three ]
legend1 = ax.legend()
plt.title("K whitening, 1 XOR")
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'KD1.svg')

fig, ax = get_subplot()
data_set_four = [base, K_i1_D2, K_i2_D2, K_i1_D2_X, K_i2_D2_X]
lines = [ ax.plot(d.x, alpha=d.alpha, label=d.label, color=colors[i]) for i,d in enumerate(data_set_four)  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_four ]
legend1 = ax.legend()
plt.title("K whitening, 2 XOR")
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'KD2.svg')
