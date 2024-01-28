import sys
sys.path.append("../")
import helper_funcs as helper
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np
from scipy.stats import ttest_ind

word_size = 8
record_size = 100000
dof = 256
prob = [2**(-1*word_size) for _ in range(dof)]


class dset():
        def __init__(self, fname, plot_label):
            self.label = plot_label
            self.fname = fname
            self.data = np.load(fname)
            self.chisq = funcs.compute_chi_squared(self.data / record_size, word_size)
            self.chisq_label = f"{self.chisq:.2f}"
            self.p_val = ttest_ind(self.chisq, prob, permutations=256).pvalue
            self.p_label = f"{self.p_val:.3f}"
            self.uniformity = funcs.get_uniformity(self.data, word_size, record_size)
            self.x = self.uniformity / record_size


def get_subplot():
    fig, ax = plt.subplots()
    ax.set_xlim( [0,255] )
    ax.set_xlabel( 'Generated 8-bit Number' )
    ax.set_ylabel( 'p' )
    return fig, ax


base       = dset('./base_freq.npy',                            "base")
print(base.p_val)
T_290_D1   = dset("./wordstreams/ws_17:07:16/no_xor.npy",       "290K, No XOR")
T_310_D1   = dset("./wordstreams/ws_17:07:44/no_xor.npy",       "310K, No XOR")
T_290_D1_X = dset("./wordstreams/ws_17:07:16/XORd_1_stream.npy","290K, One XOR")
T_310_D1_X = dset("./wordstreams/ws_17:07:44/XORd_1_stream.npy","310K, One XOR")

T_290_D2   = dset("./wordstreams/ws_17:08:24/no_xor.npy",       "290K, No XOR")
T_310_D2   = dset("./wordstreams/ws_17:08:47/no_xor.npy",       "310K, No XOR")
T_290_D2_X = dset("./wordstreams/ws_17:08:24/XORd_2_stream.npy","290K, Two XORs")
T_310_D2_X = dset("./wordstreams/ws_17:08:47/XORd_2_stream.npy","310K, Two XORs")

K_i1_D1    = dset("./wordstreams/ws_17:59:17/no_xor.npy",       "(1) K 2.5%, No XOR")
K_i2_D1    = dset("./wordstreams/ws_17:59:26/no_xor.npy",       "(2) K 2.5%, No XOR")
K_i1_D1_X  = dset("./wordstreams/ws_17:59:17/XORd_1_stream.npy","(1) K 2.5%, One XOR")
K_i2_D1_X  = dset("./wordstreams/ws_17:59:26/XORd_1_stream.npy","(2) K 2.5%, One XOR")

K_i1_D2    = dset("./wordstreams/ws_17:59:57/no_xor.npy",        "(1) K 2.5%, No XOR")
K_i2_D2    = dset("./wordstreams/ws_18:00:15/no_xor.npy",        "(2) K 2.5%, No XOR")
K_i1_D2_X  = dset("./wordstreams/ws_17:59:57/XORd_2_stream.npy", "(1) K 2.5%, Two XORs")
K_i2_D2_X  = dset("./wordstreams/ws_18:00:15/XORd_2_stream.npy", "(2) K 2.5%, Two XORs")



fig, ax = get_subplot()
data_set_one = [base, T_290_D1, T_310_D1, T_290_D1_X, T_310_D1_X]
lines = [ ax.plot(d.x, alpha=0.7, label=d.label) for d in data_set_one  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_one ]
legend1 = ax.legend()
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'./TD1.svg')

fig, ax = get_subplot()
data_set_two = [base, T_290_D2, T_290_D2_X, T_310_D2, T_310_D2_X]
lines = [ ax.plot(d.x, alpha=0.7, label=d.label) for d in data_set_two  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_two ]
legend1 = ax.legend()
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'TD2.svg')

fig, ax = get_subplot()
data_set_three = [base, K_i1_D1, K_i1_D1_X, K_i2_D1, K_i2_D1_X]
lines = [ ax.plot(d.x, alpha=0.7, label=d.label) for d in data_set_three  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_three ]
legend1 = ax.legend()
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'KD1.svg')

fig, ax = get_subplot()
data_set_four = [base, K_i1_D2, K_i1_D2_X, K_i2_D2, K_i2_D2_X]
lines = [ ax.plot(d.x, alpha=0.7, label=d.label) for d in data_set_three  ]
stats = [ d.chisq_label + ' | ' + d.p_label for d in data_set_three ]
legend1 = ax.legend()
plt.legend([l[0] for l in lines], stats, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'KD2.svg')
