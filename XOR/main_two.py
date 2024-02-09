import sys
sys.path.append('../')

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import glob

from mtj_types_v3 import SWrite_MTJ_rng
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

    length = 1000
    kdev   = 0.0
    T      = 300
    n_bins = 100
    Temps  = [290, 300, 310, 320, 330]


    #gen_x_bins(n_bins, T, kdev, length, method, out_dir, depth)
    gen_bin_T_sweep(n_bins, Temps, length, "NO", out_dir, None, 0)
    gen_bin_T_sweep(n_bins, Temps, length, "2S1D", out_dir, 1, 1)
    gen_bin_T_sweep(n_bins, Temps, length, "2S1D", out_dir, 2, 2)

    print("--- %s seconds ---" % (time.time() - start_time))


# generates x bin data with given configuration.
def gen_x_bins(x, T, kdev, length, method, out_dir, depth = None, iteration = None):
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
        probs.append( np.sum(stream)/length )

    np.savez(f"{out_dir}/metadata_{iteration}.npz",
             T = T, kdev=kdev, word_size=word_size,
             length=length, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_{T}_streamdata_{iteration}.npz",
             probs=probs)

    return probs

def gen_bin_T_sweep(x, Temps, length, method, out_dir, depth = None, iteration = None):

    probs_per_temp = []
    for T in Temps:
        probs = gen_x_bins(x, T, 0, length, method, out_dir, depth)
        probs_per_temp.append( np.average( probs ) )

    np.savez(f"{out_dir}/metadata_{iteration}.npz",
             Temps = Temps, word_size=word_size,
             length=length, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_{T}_Sweep_{iteration}.npz",
             probs_per_temp = probs_per_temp)

    return probs_per_temp



def get_wordstream_with_XOR(generator, devs, args, depth, out_dir, iteration = None):
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
    if iteration == None:
        np.save(out_dir + f'/XORd_stream.npy', XORd)
    else:
        np.save(out_dir + f'/XORd_stream_{iteration}.npy', XORd)
    '''
    return XORd



#  ========== different methods of generating an XORd bitstream ===========

def no_xor(T, kdev, length, out_dir, iteration = None):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals() #default device parameters are now updated to be NYU dev
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    if iteration == None:
        funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + '/stream.npy')
        stream = np.load(out_dir + f'/stream.npy')
    else:
        funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + f'/stream_{i}.npy')
        stream = np.load(out_dir + f'/stream_{i}.npy')

    return stream

def one_stream_split(T, kdev, length, out_dir, iteration = None):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals() #default device parameters are now updated to be NYU dev
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + '/full')

    # manually build a tree with nodes as the generated stream split in half
    root = tree.node(None)
    full = np.load(out_dir + '/full')
    # LENGTH SHOULD BE EVEN
    np.savetxt('L', full[0: (length/2)-1])
    np.savetxt('R', full[length/2 : length-1])
    root.left = tree.node('L')
    root.right = tree.node('R')
    XORd = funcs.recursive_XOR(root)

    '''
    if iteration == None:
        np.save(out_dir + f'/XORd_stream.npy', XORd)
    else:
        np.save(out_dir + f'/XORd_stream_{iteration}.npy', XORd)
    '''
    return XORd

def two_stream_one_dev(T, kdev, length, depth, out_dir, iteration = None):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals()
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev, copy.deepcopy(dev)), (V_50, word_size, length), depth, out_dir, iteration)
    return XORd

def two_stream_two_dev(T, kdev, length, depth, out_dir, iteration = None):
    dev_L = SWrite_MTJ_rng("NYU")
    dev_L.set_vals()
    dev_L.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    dev_R = SWrite_MTJ_rng("NYU")
    dev_R.set_vals()
    dev_R.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev_L, dev_R), (V_50, word_size, length), depth, out_dir, iteration)
    return XORd

if __name__ == "__main__":
    main()
