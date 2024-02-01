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

from scipy.stats import chi2
V_50 = funcs.p_to_V(0.5)
word_size = 8

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
    '''

    length = 100000
    depth  = 1
    kdev   = 0.0
    T      = 300
    method = "2S1D"

    gen_K_uniformity(kdev, length, depth, method, out_dir)
    #gen_K_p_measure(length, depth, method, out_dir)

    print("--- %s seconds ---" % (time.time() - start_time))


# K is 'constant' here
# generates uniformity data with given configuration.
def gen_T_uniformity(T, length, depth, method, out_dir):
    if method == "2S1D":
        XORd = two_stream_one_dev(T, 0, length, depth, out_dir)
    elif method == "2S2D":
        XORd = two_stream_two_dev(T, 0, length, depth, out_dir)
    elif method == "OSS":
        XORd = one_stream_split(T, 0, length, out_dir)

    L1 = np.load(out_dir + '/h1_L_0.npy')
    R1 = np.load(out_dir + '/h1_R_1.npy')
    np.savez(f"{out_dir}/metadata.npz",
             T = T, depth = depth, method = method)

    uniformity, chisq, p_val = funcs.get_stats(XORd, length)
    np.savez(f"{out_dir}/plottable_{T}_XOR.npz",
             chisq = chisq, p_val = p_val, x = uniformity/length)

    uniformity, chisq, p_val = funcs.get_stats(L1, length)
    np.savez(f"{out_dir}/plottable_{T}_L1.npz",
             chisq = chisq, p_val = p_val, x = uniformity/length)

    uniformity, chisq, p_val = funcs.get_stats(R1, length)
    np.savez(f"{out_dir}/plottable_{T}_R1.npz",
             chisq = chisq, p_val = p_val, x = uniformity/length)

    if depth == 2:
        pass
        ''' FIXME
        L2 = np.load(out_dir + '/h1_L_2.npy')
        R2 = np.load(out_dir + '/h1_R_3.npy')
        uniformity, chisq, p_val = stats(L2, length)
        np.savez(f"{out_dir}/plottable_{T}_L2.npz",
                 chisq = chisq, p_val = p_val, x = uniformity/length)

        uniformity, chisq, p_val = stats(R1, length)
        np.savez(f"{out_dir}/plottable_{T}_R2.npz",
                 chisq = chisq, p_val = p_val, x = uniformity/length)
        '''
    return

# using p value of p value distribution across multiple runs
# of given configuration
def gen_T_p_measure(length, depth, method, out_dir):

    Temps = [300, 315, 330]
    iters = 100
    p_exp = 0.05

    if method == "2S1D":
        function = two_stream_one_dev
        args = (0, length, depth, out_dir)
    elif method == "2S2D":
        function = two_stream_two_dev
        args = (0, length, depth, out_dir)
    elif method == "OSS":
        function  = one_stream_split
        args = (0, length, out_dir)

    # not bothering to save xord streams anymore...

    # for range of temperature, generate x number of 
    # p values for each bitstream uniformity
    # and then take the p value of the p values
    x_axis = []
    for T in Temps:
        for i in range(iters):
            function(T, *args, i)
        xord_streams = glob.glob(out_dir + '/XORd_stream_*')
        p_vals = [ funcs.get_stats(np.load(stream), length)[2] for stream in xord_streams ]

        # compute meta stats
        #E = 0.05
        #chisq = np.sum( [ ((O_i-E)**2)/E for O_i in p_vals ] )
        #meta_p = chi2.sf(chisq, 1) #??? dof FIXME FIXME FIXME
        meta_p = np.mean(p_vals)
        x_axis.append(meta_p)

    np.savez(f"{out_dir}/metadata.npz",
             Temps = Temps, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_{T}_Sweep.npz",
             x = x_axis)

    return

#T is constant here
#Generates uniformity data for given device configuration
def gen_K_uniformity(Kdev, length, depth, method, out_dir):
    if method == "2S1D":
        XORd = two_stream_one_dev(300, Kdev, length, depth, out_dir)
    elif method == "2S2D":
        XORd = two_stream_two_dev(300, Kdev, length, depth, out_dir)
    elif method == "OSS":
        XORd = one_stream_split(300, Kdev, length, out_dir)

    L1 = np.load(out_dir + '/h1_L_0.npy')
    R1 = np.load(out_dir + '/h1_R_1.npy')
    np.savez(f"{out_dir}/metadata.npz",
             Kdev = Kdev, depth = depth, method = method)

    uniformity, chisq, p_val = funcs.get_stats(XORd, length)
    np.savez(f"{out_dir}/plottable_{Kdev}_XOR.npz",
             chisq = chisq, p_val = p_val, x = uniformity/length)

    uniformity, chisq, p_val = funcs.get_stats(L1, length)
    np.savez(f"{out_dir}/plottable_{Kdev}_L1.npz",
             chisq = chisq, p_val = p_val, x = uniformity/length)

    uniformity, chisq, p_val = funcs.get_stats(R1, length)
    np.savez(f"{out_dir}/plottable_{Kdev}_R1.npz",
             chisq = chisq, p_val = p_val, x = uniformity/length)

    if depth == 2:
        pass
        ''' FIXME
        L2 = np.load(out_dir + '/h1_L_2.npy')
        R2 = np.load(out_dir + '/h1_R_3.npy')
        uniformity, chisq, p_val = stats(L2, length)
        np.savez(f"{out_dir}/plottable_{T}_L2.npz",
                 chisq = chisq, p_val = p_val, x = uniformity/length)

        uniformity, chisq, p_val = stats(R1, length)
        np.savez(f"{out_dir}/plottable_{T}_R2.npz",
                 chisq = chisq, p_val = p_val, x = uniformity/length)
        '''
    return

#Same operation as gen_T_p_measure, but varying K deviation
def gen_K_p_measure(length, depth, method, out_dir):

    Kdevs = [1.0, 2.5, 5.0]
    iters = 100
    p_exp = 0.05

    if method == "2S1D":
        function = two_stream_one_dev
        args = (length, depth, out_dir)
    elif method == "2S2D":
        function = two_stream_two_dev
        args = (length, depth, out_dir)
    elif method == "OSS":
        function  = one_stream_split
        args = (length, out_dir)

    # not bothering to save xord streams anymore...

    # for range of temperature, generate x number of
    # p values for each bitstream uniformity
    # and then take the p value of the p values
    x_axis = []
    for K in Kdevs:
        for i in range(iters):
            function(300, K, *args, i)
        xord_streams = glob.glob(out_dir + '/XORd_stream_*')
        p_vals = [ funcs.get_stats(np.load(stream), length)[2] for stream in xord_streams ]

        # compute meta stats
        #E = 0.05
        #chisq = np.sum( [ ((O_i-E)**2)/E for O_i in p_vals ] )
        #meta_p = chi2.sf(chisq, 1) #??? dof FIXME FIXME FIXME
        meta_p = np.mean(p_vals)
        x_axis.append(meta_p)

    np.savez(f"{out_dir}/metadata.npz",
             Kdevs = Kdevs, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_{K}_Sweep.npz",
             x = x_axis)

    return

    pass




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

    if iteration == None:
        np.save(out_dir + f'/XORd_stream.npy', XORd)
    else:
        np.save(out_dir + f'/XORd_stream_{iteration}.npy', XORd)
    return XORd





#  ========== different methods of generating an XORd bitstream ===========
def one_stream_split(T, kdev, length, out_dir, iteration = None):
    dev = SWrite_MTJ_rng()
    dev.set_vals(0) #default device parameters are now updated to be NYU dev
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    funcs.gen_wordstream(dev, length, out_dir + '/full')

    # manually build a tree with nodes as the generated stream split in half
    root = tree.node(None)
    full = np.load(out_dir + '/full')
    # LENGTH SHOULD BE EVEN
    np.savetxt('L', full[0: (length/2)-1])
    np.savetxt('R', full[length/2 : length-1])
    root.left = tree.node('L')
    root.right = tree.node('R')
    XORd = funcs.recursive_XOR(root)

    if iteration == None:
        np.save(out_dir + f'/XORd_stream.npy', XORd)
    else:
        np.save(out_dir + f'/XORd_stream_{iteration}.npy', XORd)
    return XORd

def two_stream_one_dev(T, kdev, length, depth, out_dir, iteration = None):
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev, copy.deepcopy(dev)), (V_50, word_size, length), depth, out_dir, iteration)
    return XORd

def two_stream_two_dev(T, kdev, length, depth, out_dir, iteration = None):
    dev_L = SWrite_MTJ_rng()
    dev_L.set_vals(0)
    dev_L.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    dev_R = SWrite_MTJ_rng()
    dev_R.set_vals(0)
    dev_R.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev_L, dev_R), (V_50, word_size, length), depth, out_dir, iteration)
    return XORd

if __name__ == "__main__":
    main()
