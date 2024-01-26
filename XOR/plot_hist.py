import sys
sys.path.append("../")
import helper_funcs as helper
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np


word_size = 8
base_dev = np.load("./wordstreams/ws_08:49:59/p_05.npy")

T_295_D1 = np.load("./wordstreams/ws_08:07:46/XORd_1_stream.npy")
print(base_dev)
T_295_D2 = np.load("./wordstreams/ws_08:08:52/XORd_2_stream.npy")
T_305_D1 = np.load("./wordstreams/ws_08:08:20/XORd_1_stream.npy")
T_305_D2 = np.load("./wordstreams/ws_08:09:25/XORd_2_stream.npy")

K_i1_D1 = np.load("./wordstreams/ws_07:57:25/XORd_1_stream.npy")
K_i1_D2 = np.load("./wordstreams/ws_07:58:23/XORd_2_stream.npy")
K_i2_D1 = np.load("./wordstreams/ws_07:57:49/XORd_1_stream.npy")
K_i2_D2 = np.load("./wordstreams/ws_07:58:50/XORd_2_stream.npy")



fig, ax = helper.plot_init()
ax.plot(base_dev, alpha=0.5, label="base")

#ax.plot(T_295_D1, alpha=1, color="orange", label="T=295, 1 XOR")
#ax.plot(T_305_D1, alpha=1, color="green",label="T=305, 1 XOR")
ax.set_xlim( [0,255] )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig, './T_D1.svg')

fig, ax = helper.plot_init()
#ax.plot(base_dev, alpha=0.5, label="base")
#ax.plot(T_295_D2, alpha=1, color="orange", label="T=295, 2 XOR")
#ax.plot(T_305_D2, alpha=1, color="green",label="T=305, 2 XOR")
ax.set_xlim( [0,255] )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig, './T_D2.svg')

fig, ax = helper.plot_init()
#ax.plot(base_dev, alpha=0.5, label="base")
#ax.plot(K_i1_D1, alpha=1, color="orange", label="K 2.5%, 1 XOR")
#ax.plot(K_i2_D1, alpha=1, color="green",label="K 2.5%, 1 XOR")
ax.set_xlim( [0,255] )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig, './K_D1.svg')

fig, ax = helper.plot_init()
#ax.plot(base_dev, alpha=0.5, label="base")
#ax.plot(K_i1_D2, alpha=1, color="orange", label="K 2.5%, 2 XOR")
#ax.plot(K_i2_D2, alpha=1, color="green",label="K 2.5%, 2 XOR")
ax.set_xlim( [0,255] )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig, './K_D2.svg')
'''
path = sys.argv[1] + '/'
word_size = int(sys.argv[2])
f_50 = path + 'words_50.txt'
f1 = path + 'words_1.txt'
f2 = path + 'words_2.txt'

h_50 = funcs.get_uniformity(f_50, word_size)
h1 = funcs.get_uniformity(f1, word_size)
h2 = funcs.get_uniformity(f2, word_size)

fig, ax = helper.plot_init()
ax.plot(h_50, alpha=1, label="p = 0.5")
ax.plot(h1, alpha=0.6, color="orange", label="p = 0.4")
ax.plot(h2, alpha=0.6, color="green",label="p = 0.6")
ax.set_xlim( [0,255] )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig,path + 'hist.svg')
'''
