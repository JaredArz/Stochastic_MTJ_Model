import sys
import os
sys.path.append('../')
import numpy as np
import helper_funcs as helper

import XOR_funcs as funcs
from mtj_types_v3 import SWrite_MTJ_rng

def compute_V_range():
    # if another device is used, this hardcoded J range will need to be calculated instead
    # V50% is a function of known variables and V_cutoff where V_cutoff is solvable.
    # see Rehm papers.
    return np.linspace(-0.979122, -0.43089, 1000)

def compute_weights(dev, V_range):
    samples_to_avg = 10000
    # initializes, should be run at start
    dev.set_mag_vector()
    return [ helper.avg_weight_across_samples(dev, V, samples_to_avg) for V in V_range ]

dev = SWrite_MTJ_rng()
dev.set_vals(0) #default device parameters are now updated to be NYU dev
print("Computing V range")
V_range = compute_V_range()
print("Computing ps")
ps = compute_weights(dev, V_range)
np.save('V_range.npy', V_range)
np.save('ps.npy', ps)

print("Generating wordstream")
V_50 = funcs.p_to_V(0.5)
funcs.gen_wordstream(dev, V_50, 8, 100000, './temp.npy')
ws = np.load('./temp.npy')
os.remove('./temp.npy')

uniformity = funcs.get_uniformity(ws, 8, 100000)
chisq = funcs.compute_chi_squared(uniformity, 8, 100000)
p_val = funcs.p_val(chisq)
np.savez(f"./base_uniformity.npz",
         chisq = chisq, p_val = p_val, x = uniformity/100000)
