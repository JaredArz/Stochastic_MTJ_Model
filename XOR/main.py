import sys
sys.path.append('../')

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import glob

from mtj_types import SWrite_MTJ_rng
from interface_funcs import mtj_sample
import XOR_funcs as funcs
import tree


V_50 = funcs.p_to_V(0.5)
# GLOBAL GLOBAL GLOBAL
word_size = 1

def main():
    start_time = time.time()

    out_dir = funcs.get_out_path()
    print("output dir:")
    print(out_dir)
    print("============")

    ''' internal labeling guide
    two streams, one dev: 2S1D
    two streams, two dev: 2S2D
    one stream split :    OSS
    no xor :              NO
    '''

    length = 500
    n_bins = 500
    kdev   = 0.0
    T      = 300
    Temps  = [290, 300, 310, 320, 330]
    #kdevs  = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    kdevs  = [0.01, 0.02, 0.03, 0.04, 0.05]

    #gen_x_bins(n_bins, T, kdev, length, "NO", out_dir, True)
    #gen_bin_T_sweep(n_bins, Temps, length, "NO", out_dir,   0)
    #gen_bin_T_sweep(n_bins, Temps, length, "2S2D", out_dir, 1)
    #gen_bin_T_sweep(n_bins, Temps, length, "2S2D", out_dir, 2)
    gen_bin_kdev_sweep(n_bins, kdevs, length, "NO", out_dir,   0)
    gen_bin_kdev_sweep(n_bins, kdevs, length, "2S1D", out_dir, 1)
    gen_bin_kdev_sweep(n_bins, kdevs, length, "2S1D", out_dir, 2)

    np.savez(f"{out_dir}/metadata_kdev_sweep.npz",
             kdevs = kdevs, word_size = word_size,
             length = length)
    print("--- %s seconds ---" % (time.time() - start_time))


# generates x bin data with given configuration.
def gen_x_bins(x, T, kdev, length, method, out_dir, depth, plotting_flag):
    if method == "2S1D":
        func = two_stream_one_dev
        args = (T, kdev, length, depth, out_dir)
    elif method == "2S2D":
        func = two_stream_two_dev
        args = (T, kdev, length, depth, out_dir)
    elif method == "OSS":
        func  = one_stream_split
        args = (T, kdev, length, out_dir)
    elif method == "NO":
        func =  no_xor
        args = (T, kdev, length, out_dir)
    else:
        print("invalid method")
        exit()

    probs = []
    for i in range(x):
        stream = func(*args)
        if method == "OSS":
            probs.append( np.sum(stream)/(length/2) )
        else:
            probs.append( np.sum(stream)/length )

    if plotting_flag:
        np.savez(f"{out_dir}/metadata_single_plot.npz",
                 T = T, kdev=kdev, word_size=word_size,
                 length=length, depth = depth, method = method)
        np.savez(f"{out_dir}/plottable_{T}_{kdev}_streamdata_single_plot.npz",
                 probs=probs)
    return probs

def gen_bin_T_sweep(x, Temps, length, method, out_dir, depth):
    probs_per_temp = []
    for T in Temps:
        probs = gen_x_bins(x, T, 0, length, method, out_dir, depth, False)
        probs_per_temp.append( np.average( probs ) )

    np.savez(f"{out_dir}/metadata_sweep_{depth}.npz",
             Temps = Temps, word_size=word_size,
             length=length, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_{T}_Sweep_{depth}.npz",
             probs_per_temp = probs_per_temp)

    return probs_per_temp

def gen_bin_kdev_sweep(x, kdevs, length, method, out_dir, depth):

    probs_per_kdev = []
    for kdev in kdevs:
        probs = gen_x_bins(x, 300, kdev, length, method, out_dir, depth, False)
        probs_per_kdev.append( np.average( probs ) )


    np.savez(f"{out_dir}/plottable_kdev_sweep_{depth}.npz",
             probs_per_kdev = probs_per_kdev, method = method,
             depth = depth)

    return probs_per_kdev


def get_wordstream_with_XOR(generator, devs, args, depth, out_dir):
    if out_dir is None:
        return

    # depth corresponds to representation as a binary expression tree
    # (root is at height zero)
    # ex: 2 will be a binary tree with two levels where two pairs of
    # bitstreams are xord then the two results xord
    root = tree.node(None)
    tree.build_tree(generator, devs, args, root, depth, out_dir)

    XORd = funcs.recursive_XOR(root)

    '''
    np.save(out_dir + f'/XORd_stream.npy', XORd)
    '''
    return XORd



#  ========== different methods of generating an XORd bitstream ===========

def no_xor(T, kdev, length, out_dir):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals() #default device parameters are now updated to be NYU dev
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + '/stream.npy')
    stream = np.load(out_dir + f'/stream.npy')
    return stream

def one_stream_split(T, kdev, length, out_dir):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals() #default device parameters are now updated to be NYU dev
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + '/full.npy')

    # manually build a tree with nodes as the generated stream split in half
    root = tree.node(None)
    full = np.load(out_dir + '/full.npy')
    # LENGTH SHOULD BE EVEN
    np.save('L.npy', full[0: (length//2)-1])
    np.save('R.npy', full[length//2 : length-1])
    root.left = tree.node('L.npy')
    root.right = tree.node('R.npy')
    XORd = funcs.recursive_XOR(root)

    '''
    np.save(out_dir + f'/XORd_stream.npy', XORd)
    '''
    return XORd

def two_stream_one_dev(T, kdev, length, depth, out_dir):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals()
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev, copy.deepcopy(dev)), (V_50, word_size, length), depth, out_dir)
    return XORd

def two_stream_two_dev(T, kdev, length, depth, out_dir):
    dev_L = SWrite_MTJ_rng("NYU")
    dev_L.set_vals()
    dev_L.set_vals(K_295 = dev_L.K_295 * np.random.normal(1,kdev), T = T)

    dev_R = SWrite_MTJ_rng("NYU")
    dev_R.set_vals()
    dev_R.set_vals(K_295 = dev_R.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev_L, dev_R), (V_50, word_size, length), depth, out_dir)
    return XORd

if __name__ == "__main__":
    main()
