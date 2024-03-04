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
import os


V_50 = funcs.p_to_V(0.5)
# GLOBAL GLOBAL GLOBAL
word_size = 1

def main():
    start_time = time.time()

    pid = os.getpid()
    pidfloc = 'PIDfiles/'
    pidfilename = pidfloc + str(pid) + '.txt'
    # this is to clear old names out of the folder

    os.system('rm ' + pidfloc + '*.txt >/dev/null 2>&1')
    time.sleep(2)

    pf = open(pidfilename,'w')
    pf.write(str(pid))
    pf.close

    time.sleep(2)

    pidfilelist = os.listdir(pidfloc)

    numpids = 0
    for filename in pidfilelist:
        numpids += 1

    pidList = np.zeros(numpids)
    fn = 0
    for filename in pidfilelist:
        a = filename[:-4]
        pidList[fn] = float(a)
        if(pidList[fn] == pid):
            print(str(pid) + ' assigned to local rank ' + str(fn))
            run = fn
        fn += 1

    if run > 37:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run < 37 and run > 32:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run < 32 and run > 29:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run == 28:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run == 26:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run == 23:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run == 20:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run < 19 and run > 6:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return
    if run < 5 and run > 2:
        prstr = str(run) + ' doesnt have anymore work to do!'
        print(prstr)
        return

    out_dir = f"/scratch/06859/ajmaicke/wordstreams/ws_{run}"
    print("output dir:")
    print(out_dir)
    print("============")
    os.system('mkdir ' + out_dir)

    ''' internal labeling guide
    two streams, one dev: 2S1D
    two streams, two dev: 2S2D
    one stream split :    OSS
    no xor :              NO
    '''

    length = 100000
    n_bins = 10000
    kdev   = 0.0
    T      = 300

    T_arr = [290, 295, 300, 305, 310]
    k_arr = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    tid,kid = funcs.get_arr_inds(run)
    Temps  = T_arr[tid]
    kdevs  = k_arr[kid]

    prstr = 'pid: ' + str(run) + ' temp: ' + str(Temps) + ' kdev: ' + str(kdevs)

    if run < 10:
        prstr = prstr + ' dev: NO\n'
        print(prstr)
        gen_bin_T_sweep(n_bins, Temps, length, "NO", out_dir,   0)
        
    elif run < 32:
        prstr = prstr + ' dev: NO\n'
        print(prstr)
        gen_bin_kdev_sweep(n_bins, kdevs, length, "NO", out_dir,   0)
    else:
        kdevs = k_arr[run-32]
        prstr = 'pid: ' + str(run) + ' temp: ' + str(Temps) + ' kdev: ' + str(kdevs)
        prstr = prstr + ' dev: bucket NO\n'
        print(prstr)
        gen_bin_kdev_sweep(n_bins, kdevs, length, "BNO", out_dir,   0)

    #np.savez(f"{out_dir}/metadata_sweep.npz",
    #         Temps = Temps, kdevs = kdevs, word_size = word_size,
    #         length = length)
    print("--- %s seconds ---" % (time.time() - start_time))


# generates x bin data with given configuration.
def gen_x_bins(x, T, kdev, length, method, out_dir, depth, plotting_flag):
    if method == "2S1D":
        func = two_stream_one_dev
        dev = SWrite_MTJ_rng("NYU")
        dev.set_vals() #default device parameters are now updated to be NYU dev
        dev.set_vals(K_295 = dev.K_295 * (1+kdev), T = T)
        args = (T, kdev, length, depth, out_dir, dev)
    elif method == "2S2D":
        func = two_stream_two_dev
        dev_L = SWrite_MTJ_rng("NYU")
        dev_L.set_vals()
        dev_L.set_vals(K_295 = dev_L.K_295 * (1+kdev), T = T)
        dev_R = SWrite_MTJ_rng("NYU")
        dev_R.set_vals()
        dev_R.set_vals(K_295 = dev_R.K_295 * (1+kdev), T = T)
        args = (T, kdev, length, depth, out_dir, dev_L, dev_R)
    elif method == "OSS":
        func  = one_stream_split
        dev = SWrite_MTJ_rng("NYU")
        dev.set_vals() #default device parameters are now updated to be NYU dev
        dev.set_vals(K_295 = dev.K_295 * (1+kdev), T = T)
        args = (T, kdev, length, out_dir,dev)
    elif method == "NO":
        func =  no_xor
        dev = SWrite_MTJ_rng("NYU")
        dev.set_vals() #default device parameters are now updated to be NYU dev
        dev.set_vals(K_295 = dev.K_295 * (1+kdev), T = T)
        args = (T, kdev, length, out_dir,dev)
    elif method == "BNO":
        func =  binned_no_xor
        dev = SWrite_MTJ_rng("NYU")
        dev.set_vals() #default device parameters are now updated to be NYU dev
        dev.set_vals(T = T)
        args = (T, kdev, length, out_dir,dev)

    else:
        print("invalid method")
        exit()

    probs = []
    streams = []
    for i in range(x):
        stream = func(*args)
        if method == "OSS":
            probs.append( np.sum(stream)/(length/2) )
        else:
            probs.append( np.sum(stream)/length )
            streams.append(stream)

    if plotting_flag:
        np.savez(f"{out_dir}/metadata_stream.npz",
                 T = T, kdev=kdev, word_size=word_size,
                 length=length, depth = depth, method = method)
        np.savez(f"{out_dir}/plottable_streamdata.npz",
                 streams=streams)
    return probs

def gen_bin_T_sweep(x, Temps, length, method, out_dir, depth):
    probs_per_temp = []
    std_per_temp = []
    #for T in Temps:
    probs = gen_x_bins(x, Temps, 0, length, method, out_dir, depth, True)
    probs_per_temp.append( np.average( probs ) )
    std_per_temp.append( np.std( probs ) )

    np.savez(f"{out_dir}/metadata_sweep.npz",
             Temps = Temps, word_size=word_size,
             length=length, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_T_Sweep.npz",
             probs_per_temp = probs_per_temp, method = method,
             depth = depth, std_per_temp = std_per_temp)

    return probs_per_temp

def gen_bin_kdev_sweep(x, kdevs, length, method, out_dir, depth):

    probs_per_kdev = []
    std_per_kdev = []
    #for kdev in kdevs:
    probs = gen_x_bins(x, 300, kdevs, length, method, out_dir, depth, True)
    probs_per_kdev.append( np.average( probs ) )
    std_per_kdev.append( np.std( probs ) )

    np.savez(f"{out_dir}/metadata_sweep.npz",
             kdevs = kdevs, word_size=word_size,
             length=length, depth = depth, method = method)

    np.savez(f"{out_dir}/plottable_kdev_sweep.npz",
             probs_per_kdev = probs_per_kdev, method = method,
             depth = depth, std_per_kdev = std_per_kdev)

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

def no_xor(T, kdev, length, out_dir, dev):
    # dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + '/stream.npy')
    stream = np.load(out_dir + f'/stream.npy')
    return stream

def binned_no_xor(T, kdev, length, out_dir, dev):
    dev = SWrite_MTJ_rng("NYU")
    dev.set_vals()
    dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,abs(kdev)), T = T)

    funcs.gen_wordstream(dev, V_50, word_size, length, out_dir + '/stream.npy')
    stream = np.load(out_dir + f'/stream.npy')
    return stream


def one_stream_split(T, kdev, length, out_dir, dev):
    #dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

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

def two_stream_one_dev(T, kdev, length, depth, out_dir, dev):
    #dev.set_vals(K_295 = dev.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev, copy.deepcopy(dev)), (V_50, word_size, length), depth, out_dir)
    return XORd

def two_stream_two_dev(T, kdev, length, depth, out_dir, dev_L, dev_R):
    # dev_L.set_vals(K_295 = dev_L.K_295 * np.random.normal(1,kdev), T = T)
    #dev_R.set_vals(K_295 = dev_R.K_295 * np.random.normal(1,kdev), T = T)

    XORd = get_wordstream_with_XOR(funcs.gen_wordstream,
                            (dev_L, dev_R), (V_50, word_size, length), depth, out_dir)
    return XORd

if __name__ == "__main__":
    main()
