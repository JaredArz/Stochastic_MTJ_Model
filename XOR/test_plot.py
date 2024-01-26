import sys
sys.path.append("../")
import helper_funcs as helper
import matplotlib.pyplot as plt
import XOR_funcs as funcs
import helper_funcs
import sys

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
