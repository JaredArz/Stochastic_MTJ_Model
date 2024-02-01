import sys
sys.path.append('../')
import os
from datetime import datetime
import numpy as np
from interface_funcs import mtj_sample
from scipy.stats import chi2

import helper_funcs as helper

def XOR_op(a, b):
    return [ int(a_i) ^ int(b_i) for a_i,b_i in zip(a,b) ]

def recursive_XOR(root):
    # if root is a leaf node
    if root.left == None and root.right == None:
        return np.load(root.fname)
    return XOR_op( recursive_XOR(root.left), recursive_XOR(root.right)  )

def gen_wordstream(dev, supposed_V50, word_size, length, out_path):
    words = []
    dev.set_mag_vector()
    for _ in range(length):
        word = np.sum( [ mtj_sample(dev, supposed_V50)[0]*2**i for i in range(word_size) ] )
        words.append(word)
    np.save(out_path, words)
    return


# a LUT. requires generating v_range and p array
def p_to_V(p) -> float:
    if p < 0 or p > 1:
        print("p ∉ [0, 1]... Exiting")
        exit()
    V_range = np.load('./V_range.npy')
    ps = np.load('./ps.npy')
    return V_range[helper.find_idx_at_nearest(ps, p)]

def get_uniformity(word_stream, word_size, record_size):
    word_freq = np.zeros( 2**word_size )
    for number in word_stream:
        word_freq[ int(number) ] += 1
    return word_freq

def compute_chi_squared(O, word_size, record_size):
    E = 2**(-1*word_size) * record_size
    return np.sum( [ ((O_i-E)**2)/E for O_i in O ] )

def p_val(chisq):
    return chi2.sf(chisq, 256)

def get_stats(word_stream, length):
    uniformity = get_uniformity(word_stream, 8, length)
    chisq = compute_chi_squared(uniformity, 8, length)
    p = p_val(chisq)
    return uniformity, chisq, p


def get_out_path() -> str:
    date = datetime.now().strftime("%H:%M:%S")
    out_path = f"./wordstreams/ws_{date}"
    make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
    make_dir("./wordstreams")
    make_dir(f"{out_path}")
    return out_path
