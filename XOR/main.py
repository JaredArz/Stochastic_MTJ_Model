import sys
sys.path.append('../')

import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from mtj_types_v3 import SWrite_MTJ_rng
from interface_funcs import mtj_sample
import XOR_funcs as funcs
import tree

V_50 = funcs.p_to_V(0.5)
word_size = 8
dof = 256

def main():
    start_time = time.time()

    out_dir = funcs.get_out_path()
    print("output dir:")
    print(out_dir)
    print("============")

    ''' code for labeling
    two streams, one dev: 2S1D
    two streams, two dev: 2S2D
    one stream split :    OSS
    '''

    length = 1000
    depth  = 1
    kdev   = 0.0
    T      = 300
    method = "2S1D"

    gen_T_uniformity(T, kdev, length, depth, method, out_dir)

    print("--- %s seconds ---" % (time.time() - start_time))


def gen_T_uniformity(T, kdev, length, depth, method, out_dir):
    if method == "2S1D":
        XORd = two_stream_one_dev(T, kdev, length, depth, out_dir)
    elif method == "2S2D":
        XORd = two_stream_two_dev(T, kdev, length, depth, out_dir)
    elif method == "OSS":
        XORd = one_stream_split(T, kdev, length, out_dir)

    uniformity = funcs.get_uniformity(XORd, word_size, length)
    chisq = funcs.compute_chi_squared(uniformity, word_size, length)
    p_val = chi2.sf(chisq, dof)
    x_axis = uniformity / record_size
    np.savez(f"{out_dir}/plottable.npz",
             T=T, kdev=kdev, length=length, depth=depth, method=method\
             chisq = chisq, p_val = p_val, x = x_axis)

def gen_T_p_measure():

def gen_K_uniformity():

def gen_K_p_measure():

def one_stream_split(T, kdev, length, out_dir):
    dev = SWrite_MTJ_rng()
    dev.set_vals(0) #default device parameters are now updated to be NYU dev
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    gen_wordstream(dev, length, out_dir + '/OSS')

    # manually build a tree with nodes as the generated stream split in half
    root = tree.node(None)
    full = np.load(out_dir + '/OSS')
    # LENGTH SHOULD BE EVEN
    np.savetxt('L', full[0: (length/2)-1])
    np.savetxt('R', full[length/2 : length-1])
    left = tree.node('L')
    right = tree.node('R')
    root.left = left
    root.right = right
    XORd = recursive_XOR(root)

    np.save(out_dir + f'/XORd_stream.npy', XORd)
    #np.savez(f"{out_dir}/metadata.npz",
    #         T=T, kdev=kdev, length=length/2, method="OSS")
    return XORd

def two_stream_one_dev(T, kdev, length, depth, out_dir):
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = gen_wordstream_with_XOR(gen_wordstream,
                            (dev, copy.deepcopy(dev)), (V_50, word_size, length),
                            depth, out_dir)
    #np.savez(f"{out_dir}/metadata.npz",
    #         T=T, kdev=kdev, length=length, depth=depth, method="2S1D")
    return XORd

def two_stream_two_dev(T1, T2, kdev1, kdev2, length, depth, out_dir):
    dev_L = SWrite_MTJ_rng()
    dev_L.set_vals(0)
    dev_L.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev1), T = T1)

    dev_R = SWrite_MTJ_rng()
    dev_R.set_vals(0)
    dev_R.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev2), T = T2)

    XORd = gen_wordstream_with_XOR(gen_wordstream,
                            (dev_L, dev_R), (V_50, word_size, length),
                            depth, out_dir)
    #np.savez(f"{out_dir}/metadata.npz",
    #         T1=T1, T2=T2, kdev1=kdev1, kdev2=kdev2, length=length, depth=depth, method="2S2D")
    return XORd

def gen_wordstream_with_XOR(generator, devs, args, depth, out_dir):
    if out_dir is None:
        return
    # depth corresponds to representation as a binary expression tree
    # (root is at height zero)
    # ex: 2 will be a binary tree with two levels where two pairs of
    # bitstreams are xord then the two results xord
    root = tree.node(None)
    tree.build_tree(generator, devs, args, root, depth, out_dir)

    XORd = recursive_XOR(root)

    np.save(out_dir + f'/XORd_stream.npy', XORd)
    return XORd

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
