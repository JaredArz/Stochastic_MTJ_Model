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



def main():
    start_time = time.time()
    out_path = get_out_path()
    print("output path:")
    print(out_path)
    dev = SWrite_MTJ_rng()
    dev.set_mag_vector()
    dev.set_vals(0)
    dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha=0.016)
    T = 300

    args = (T)

    XOR_init()


    J_50 = get_J50(dev, Js, T)
    print(J_50*3.18e-12)

    #np.savez(f"{out_path}/metadata_voltage.npz",
    #         voltages=voltages, pulse_duration=t, Temps=Temps)
    print("--- %s seconds ---" % (time.time() - start_time))
    return

def calibrate_current(dev, T):
    # compute on the fly:
    # if another device is used, this hardcoded value will need to be calculated instead
    # V50% is a function of known variables and V_cutoff where V_cutoff is solvable.
    # see Rehm papers.
    Js = np.linspace(-3.079e11, -1.355e11, 5000)

    samples_to_avg = 10000
    weights = []
    for J in Js:
        weights.append(avg_weight_across_samples(dev, J, T, samples_to_avg))
    return weights, Js


def XOR(a, b):
    return np.xor(a,b)


def XOR_depth_n(n, dev, args):
    T, = args
    K, Ms = get_mk(T)
    dev.set_vals(Ki=K, Ms=Ms)


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
    make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
    #create dir and write path
    date = datetime.now().strftime("%H:%M:%S")
    out_path = f"./bitstreams/bs_{date}"
    make_dir("./bitstreams")
    make_dir(f"{out_path}")
    return out_path

def avg_weight_across_samples(dev,J,T,samples_to_avg) -> float:
    sum_of_samples = np.sum([(mtj_sample(dev,J,T=T))[0] for _ in range(samples_to_avg)])
    return sum_of_samples/samples_to_avg

def find_idx_at_nearest(arr, val) -> int:
    return (np.abs(np.asarray(arr)-val)).argmin()

def p_to_J(dev, Js, T, samples_to_avg=10000) -> float:
    weights = []
    for J in Js:
        weights.append(avg_weight_across_samples(dev, J, T, samples_to_avg))
    return Js[find_idx_at_nearest(weights, 0.5)]

def get_J50(dev, Js, T, samples_to_avg=10000) -> float:
    weights = []
    for J in Js:
        weights.append(avg_weight_across_samples(dev, J, T, samples_to_avg))
    return Js[find_idx_at_nearest(weights, 0.5)]


if __name__ == "__main__":
    main()
