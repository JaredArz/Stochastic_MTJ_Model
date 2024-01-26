import sys
sys.path.append('../')

import time
import numpy as np
import matplotlib.pyplot as plt

from mtj_types_v3 import SWrite_MTJ_rng
from interface_funcs import mtj_sample
import XOR_funcs as funcs
import tree


def main():
    start_time = time.time()
    out_dir = funcs.get_out_path()
    print("output dir:")
    print(out_dir)
    print("============")

    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    #FIXME prior to demag calculation
    dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha=0.016, RA=3.18e-12)
    dev.set_vals(Ms_295 = 165576.94999)

    V_range = funcs.compute_V_range()
    ps = funcs.compute_weights(dev, V_range)

    V_50 = funcs.p_to_V(0.5, ps, V_range)


    T = 305
    stddev = 0.0
    dev.set_vals(K_295 = (0.001161866/(2.6e-9)) * np.random.normal(1,stddev,1) )
    dev.set_vals(T=T)

    word_size = 8
    length = 100000
    depth = 2

    gen_wordstream(dev, V_50, word_size, length, out_dir + '/p_05')

    gen_wordstream_with_XOR(gen_wordstream,
                            (dev, V_50, word_size, length),
                            depth, out_dir)


    print(f"--- {(time.time() - start_time):.4} seconds ---")
    return

def gen_wordstream_with_XOR(generator, args, depth, out_dir):
    if out_dir is None:
        return
    # depth corresponds to representation as a binary expression tree
    # (root is at height zero)
    # ex: 2 will be a binary tree with two levels where two pairs of
    # bitstreams are xord then the two results xord
    root = tree.node(None)
    tree.build_tree(generator, args, root, depth, out_dir)
    tree.print_tree(root)

    XORd = recursive_XOR(root)

    np.save(out_dir + f'/XORd_{depth}_stream.npy', XORd)
    return

def recursive_XOR(root):
    if tree.is_leaf(root):
        return np.load(root.fname)
    return funcs.XOR_op( recursive_XOR(root.left), recursive_XOR(root.right)  )

def gen_wordstream(dev, supposed_V50, word_size, length, out_path):
    words = []
    dev.set_mag_vector()
    for _ in range(length):
        word = np.sum( [ mtj_sample(dev, supposed_V50)[0]*2**i for i in range(word_size) ] )
        words.append(word)

    np.save(out_path, words)
    return

if __name__ == "__main__":
    main()
