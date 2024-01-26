import sys
sys.path.append('../')
import os
from datetime import datetime
import numpy as np

import helper_funcs as helper

def XOR_op(a, b):
    c = [ int(a_i) ^ int(b_i) for a_i,b_i in zip(a,b) ]
    #return np.logical_xor(a,  b)
    return c

def p_to_V(p, weights, Vs) -> float:
    if p < 0 or p > 1:
        print("p ∉ [0, 1]... Exiting")
        exit()
    return Vs[helper.find_idx_at_nearest(weights, p)]

def get_out_path() -> str:
    date = datetime.now().strftime("%H:%M:%S")
    out_path = f"./wordstreams/ws_{date}"
    make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
    make_dir("./wordstreams")
    make_dir(f"{out_path}")
    return out_path

def get_uniformity(path_to_file, word_size):
    word_stream = np.loadtxt(path_to_file)

    word_freq = np.zeros( 2**word_size )

    for word_value in word_stream:
        word_freq[ int(word_value) ] += 1

    return word_freq

def compute_V_range():
    # if another device is used, this hardcoded J range will need to be calculated instead
    # V50% is a function of known variables and V_cutoff where V_cutoff is solvable.
    # see Rehm papers.

    V_range = np.linspace(-0.979122, -0.43089, 100)
    return V_range


def compute_weights(dev, V_range):
    samples_to_avg = 1000
    # initializes, should be run at start
    dev.set_mag_vector()
    return [ helper.avg_weight_across_samples(dev, V, samples_to_avg) for V in V_range ]
