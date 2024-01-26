import sys
sys.path.append("../")
import helper_funcs as helper
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import numpy as np


word_size=8
path = sys.argv[1] + '/'
f_50 = path + 'p_05.npy'

h_50 = funcs.get_uniformity(f_50, word_size)

fig, ax = helper.plot_init()
ax.plot(h_50, alpha=1, label="p = 0.5")
ax.set_xlim( [0,255] )
ax.set_ylim( [0,500] )
ax.legend()
helper.prompt_show()
helper.prompt_save_svg(fig,path + 'hist.svg')
