import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../fortran_source')

import os
import time
import numpy as np

from mtj_types_v3 import SWrite_MTJ_rng
from datetime import datetime

from interface_funcs import mtj_sample

import tree



def main():
    start_time = time.time()
    out_path = get_out_path()
    print("output path:")
    print(out_path)
    print("============")
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    # set constants here
    dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha=0.016, T=300)
    # define variables here
    args = dict()

    ps, Js = calibrate_current(dev)

    #J_50 = p_to_J(0.5, ps, Js)
    #print( f"J50 in volts: {J_50*3.18e-12}" )
    #XOR_test(Js, ps, 0.6, 0.4, dev, args, out_path)

    root = generate_tree(2, Js, ps, dev, args, out_path)
    tree.print_tree(root)

    print(f"--- {(time.time() - start_time):.4} seconds ---")
    return

def generate_bitstream(dev, J, x, out_path):
    b = np.array([mtj_sample(dev,J)])
    f = f"{out_path}/bs_{x}.txt"
    np.save(f, b)
    return f

def calibrate_current(dev):
    # if another device is used, this hardcoded J range will need to be calculated instead
    # V50% is a function of known variables and V_cutoff where V_cutoff is solvable.
    # see Rehm papers.

    Js = np.linspace(-3.079e11, -1.355e11, 100)
    samples_to_avg = 100
    # initializes, should be run at start
    dev.set_mag_vector()
    weights = [ avg_weight_across_samples(dev, J, samples_to_avg) for J in Js ]
    dev.set_mag_vector()

    return weights, Js

def XOR_op(a, b):
    return np.xor(a,b)

def generate_tree(n, Js, ps, dev, args, out_path):
    # depth corresponds to representation as a binary expression tree
    # ex: n=2 will be a binary tree with two levels where two pairs of 
    # bitstreams are xord then the two results xord

    for key, val in args.items():
        dev.set_vals( key=val )

    if 'T' in args.keys():
        K, Ms = get_mk(args["T"])
        dev.set_vals(Ki=K, Ms=Ms)

    dev.set_mag_vector()

    J = p_to_J(0.5, ps, Js)
    root = tree.node('root')
    tree.build_tree(generate_bitstream, dev, root, n, J, out_path)

    return root

def XOR_test(Js, ps, p1, p2, dev, args, out_path):
    for key, val in args.items():
        dev.set_vals( key=val )

    if 'T' in args.keys():
        K, Ms = get_mk(args["T"])
        dev.set_vals(Ki=K, Ms=Ms)

    dev.set_mag_vector()
    print(dev)

    #gen actual bitsream
    J_p1 = p_to_J(p1, ps, Js)
    b1 = [mtj_sample(dev,J_p1)]
    file = f"{out_path}/bs_{p1}.txt"
    np.savetxt(file,np.array(b))

    #gen actual bitsream
    J_p2 = p_to_J(p2, ps, Js)
    b2 = [mtj_sample(dev,J_p2)]
    file = f"{out_path}/bs_{p2}.txt"
    np.savetxt(file,np.array(b))


    return XOR_op(b1,b2)

def get_mk(T):
    Tc = 1453
    n = 1.804
    q = 1.0583
    Kstar = 4.389e5
    Mstar = 5.8077e5
    Ms_295 = 165576.94999 #the values that match exp, eyeballed FIXME prior to demag calculation
    K_295 = 0.001161866/(2.6e-9)
    cm = Ms_295 - Mstar*( 1 - (295/Tc)**q )
    ck = K_295 - Kstar*( ( Ms_295/Mstar )**n )
    #cm = 0
    #ck = 0

    # fitted curves with constant offset
    Ms = Mstar*( 1 - (T/Tc)**q ) + cm
    K = 2.6e-9*((Kstar)*( (Ms/Mstar)**n ) + ck)

    return (K,Ms)

def get_out_path() -> str:
    date = datetime.now().strftime("%H:%M:%S")
    out_path = f"./bitstreams/bs_{date}"
    make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
    make_dir("./bitstreams")
    make_dir(f"{out_path}")
    return out_path

def avg_weight_across_samples(dev, J, samples_to_avg) -> float:
    sum_p = np.sum( [ (mtj_sample(dev,J),) for _ in range(samples_to_avg)] )
    return sum_p/samples_to_avg

def find_idx_at_nearest(vec, val) -> int:
    vector_difference = np.abs( np.asarray(vec) - val )
    return vector_difference.argmin()

def p_to_J(p, weights, Js) -> float:
    if p < 0 or p > 1:
        print("p ∉ [0, 1]... Exiting")
        exit()
    return Js[find_idx_at_nearest(weights, p)]


if __name__ == "__main__":
    main()
