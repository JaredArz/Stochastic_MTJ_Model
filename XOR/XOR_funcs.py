import sys
sys.path.append('../')
import os
from datetime import datetime
import numpy as np

import helper_funcs as helper

def XOR_op(a, b):
    return [ int(a_i) ^ int(b_i) for a_i,b_i in zip(a,b) ]

# a LUT. requires generating v_range and p array
def p_to_V(p) -> float:
    if p < 0 or p > 1:
        print("p ∉ [0, 1]... Exiting")
        exit()
    V_range = np.load('./V_range.npy')
    ps = np.load('./ps.npy')
    return V_range[helper.find_idx_at_nearest(ps, p)]

def get_out_path() -> str:
    date = datetime.now().strftime("%H:%M:%S")
    out_path = f"./wordstreams/ws_{date}"
    make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
    make_dir("./wordstreams")
    make_dir(f"{out_path}")
    return out_path

def get_uniformity(word_stream, word_size, record_size):
    word_freq = np.zeros( 2**word_size )
    for number in word_stream:
        word_freq[ int(number) ] += 1
    return word_freq

def compute_chi_squared(norm_O, word_size):
    E = 2**(-1*word_size)
    return np.sum( [ ((O_i-E)**2)/E for O_i in norm_O ] )

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
