import sys
sys.path.append("../")
import helper_funcs as helper
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np
from scipy.stats.distributions import chi2

word_size = 8
record_size = 100000
dof = 256

class dset(){
        def __init__(fname, ):
            self.fname = fname
            data = np.load(fname)
            chisq = funcs.compute_chi_squared(base, word_size, record_size)
            chisq_label = f"{chisq}:.2f"
            p_val = f"{chi2.sf(base_chisq_label, dof)}:.2f"
            p_label = f"{chi2.sf(base_chisq_label, dof)}:.2f"
}


fT_290_D1 = "./wordstreams/ws_17:07:16/no_xor.npy"
fT_290_D2 = "./wordstreams/ws_17:08:24/no_xor.npy"
fT_310_D1 = "./wordstreams/ws_17:07:44/no_xor.npy"
fT_310_D2 = "./wordstreams/ws_17:08:47/no_xor.npy"
fT_290_D1_X = "./wordstreams/ws_17:07:16/XORd_1_stream.npy"
fT_290_D2_X = "./wordstreams/ws_17:08:24/XORd_2_stream.npy"
fT_310_D1_X = "./wordstreams/ws_17:07:44/XORd_1_stream.npy"
fT_310_D2_X = "./wordstreams/ws_17:08:47/XORd_2_stream.npy"

fK_i1_D1    = "./wordstreams/ws_17:59:17/no_xor.npy"
fK_i1_D1_X  = "./wordstreams/ws_17:59:17/XORd_1_stream.npy"
fK_i2_D1    = "./wordstreams/ws_17:59:26/no_xor.npy"
fK_i2_D1_X  = "./wordstreams/ws_17:59:26/XORd_1_stream.npy"

fK_i1_D2    = "./wordstreams/ws_17:59:57/no_xor.npy"
fK_i1_D2_X  = "./wordstreams/ws_17:59:57/XORd_2_stream.npy"
fK_i2_D2    = "./wordstreams/ws_18:00:15/no_xor.npy"
fK_i2_D2_X  = "./wordstreams/ws_18:00:15/XORd_2_stream.npy"

T_290_D1 = np.load(fT_290_D1)
T_290_D2 = np.load(fT_290_D2)
T_310_D1 = np.load(fT_310_D1)
T_310_D2 = np.load(fT_310_D2)

T_290_D1_X = np.load(fT_290_D1_X)
T_290_D2_X = np.load(fT_290_D2_X)
T_310_D1_X = np.load(fT_310_D1_X)
T_310_D2_X = np.load(fT_310_D2_X)

K_i1_D1    = np.load(fK_i1_D1  )
K_i2_D1    = np.load(fK_i2_D1  )
K_i1_D2    = np.load(fK_i1_D2 )
K_i2_D2    = np.load(fK_i2_D2)

K_i1_D1_X  = np.load(fK_i1_D1_X)
K_i1_D2_X  = np.load(fK_i1_D2_X)
K_i2_D1_X  = np.load(fK_i2_D1_X)
K_i2_D2_X  = np.load(fK_i2_D2_X)

base = np.load('./base_freq.npy')
base_chisq = funcs.compute_chi_squared(base, word_size, record_size)
base_chisq_label = f"{base_chisq}:.2f"
base_p_label = f"{chi2.sf(base_chisq_label, dof)}:.2f"

fig, ax = helper.plot_init()
A = ax.plot(funcs.get_uniformity(base, word_size)/record_size,       alpha=0.4, label=f"baseline")
B = ax.plot(funcs.get_uniformity(T_290_D1, word_size)/record_size,   alpha=0.6, label=f"290K no xor")
C = ax.plot(funcs.get_uniformity(T_290_D1_X, word_size)/record_size, alpha=0.6, label=f"290K 1 xor")
D = ax.plot(funcs.get_uniformity(T_310_D1, word_size)/record_size,   alpha=0.6, label=f"310K no xor")
E = ax.plot(funcs.get_uniformity(T_310_D1_X, word_size)/record_size, alpha=0.6, label=f"310K 1 xor")

T_290_D1_chisq = funcs.compute_chi_squared(T_290_D1,  word_size, record_size)
funcs.compute_chi_squared(T_290_D1_X,word_size, record_size)
funcs.compute_chi_squared(T_310_D1,  word_size, record_size)
funcs.compute_chi_squared(T_310_D1_X,word_size, record_size)


ax.set_xlim( [0,255] )
ax.set_xlabel( 'Generated 8-bit Number' )
ax.set_ylabel( 'p' )
chi2s = [1, 2, 3, 4]
plot_lines = [A, B, C, D]
legend1 = ax.legend()
plt.legend([l[0] for l in plot_lines], chi2s, loc=4)
plt.gca().add_artist(legend1)
helper.prompt_show()
helper.prompt_save_svg(fig,'./TD1.svg')



fig, ax = helper.plot_init()

ax.plot(funcs.get_uniformity(base, word_size), alpha=0.5, label=f"baseline {funcs.compute_chi_squared(base,word_size):.2f}")
ax.plot(funcs.get_uniformity(T_290_D2, word_size), alpha=0.6, label=f"290K no xor {funcs.compute_chi_squared(T_290_D2,word_size):.2f}")
ax.plot(funcs.get_uniformity(T_290_D2_X, word_size), alpha=0.8, label=f"290K 2 xor {funcs.compute_chi_squared(T_290_D2_X,word_size):.2f}")
ax.plot(funcs.get_uniformity(T_310_D2, word_size), alpha=0.6, label=f"310K no xor {funcs.compute_chi_squared(T_310_D2,word_size):.2f}")
ax.plot(funcs.get_uniformity(T_310_D2_X, word_size), alpha=0.8, label=f"310K 2 xor {funcs.compute_chi_squared(T_310_D2_X,word_size):.2f}")
ax.set_xlim( [0,255] )
ax.set_xlabel( 'Generated 8-bit Number' )
ax.set_ylabel( 'p' )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig,'TD2.svg')

fig, ax = helper.plot_init()

ax.plot(funcs.get_uniformity(base, word_size), alpha=0.5, label=f"baseline | {funcs.compute_chi_squared(base,word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i1_D1,  word_size), alpha=0.6, label=f"K 2.5% no xor | {funcs.compute_chi_squared(K_i1_D1, word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i1_D1_X, word_size), alpha=0.8, label=f"K 2.5% 1 xor | {funcs.compute_chi_squared(K_i1_D1_X,word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i2_D1,  word_size), alpha=0.6, label=f"K 2.5% no xor | {funcs.compute_chi_squared(K_i2_D1, word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i2_D1_X, word_size), alpha=0.8, label=f"K 2.5% 1 xor | {funcs.compute_chi_squared(K_i2_D1_X,word_size):.2f}")
ax.set_xlim( [0,255] )
ax.set_xlabel( 'Generated 8-bit Number' )
ax.set_ylabel( 'p' )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig,'KD1.svg')

fig, ax = helper.plot_init()

ax.plot(funcs.get_uniformity(base, word_size), alpha=0.5, label=f"baseline | {funcs.compute_chi_squared(base,word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i1_D2,  word_size), alpha=0.6, label=f"K 2.5% no xor | {funcs.compute_chi_squared(K_i1_D2, word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i1_D2_X, word_size), alpha=0.8, label=f"K 2.5% 2 xor | {funcs.compute_chi_squared(K_i1_D2_X,word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i2_D2,  word_size), alpha=0.6, label=f"K 2.5% no xor | {funcs.compute_chi_squared(K_i2_D2, word_size):.2f}")
ax.plot(funcs.get_uniformity(K_i2_D2_X, word_size), alpha=0.8, label=f"K 2.5% 2 xor | {funcs.compute_chi_squared(K_i2_D2_X,word_size):.2f}")
ax.set_xlim( [0,255] )
ax.set_xlabel( 'Generated 8-bit Number' )
ax.set_ylabel( 'p' )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig,'KD2.svg')
