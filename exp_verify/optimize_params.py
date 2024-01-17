import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../fortran_source')
import numpy as np
import glob

from mtj_types_v3 import SWrite_MTJ_rng
import helper_exp_verify as helper

#XXX use at own risk, no guarantee of correctness


T = 300
RA = 3.18e-12

pulse_durations = np.loadtxt('./exp_data/fig1aLR.txt',usecols=0)
t_weights = np.loadtxt('./exp_data/fig1aLR.txt',usecols=1)
voltages = np.flip([ -v for v in np.loadtxt('./exp_data/fig1bLR300.txt',usecols=0)])
v_weights = np.loadtxt('./exp_data/fig1bLR300.txt',usecols=1)
mean_v_weight = np.mean(v_weights)
v_weights[0] = 1e-3
t_weights[0] = 1e-3

dev = SWrite_MTJ_rng()
dev.set_vals(0)
dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha = 0.016)

def main():

    K_range = np.linspace(0.0011, 0.0015, 50)
    Ms_range = np.linspace(165000, 166000, 50)

    result = search(error_function, K_range, Ms_range)
    f = open("opt.txt", 'w')
    f.write(str(result))
    f.close()

def search(f, K_range, Ms_range):
    min_e = 1e8
    best_pair = None
    for K_i in K_range:
        for M_i in Ms_range:
            e = f(K_i, M_i)
            if e < min_e:
                min_e = e
                best_pair = (K_i, M_i)
    return (best_pair, min_e)

def error_function(K, Ms):
    dev.set_mag_vector()
    dev.set_vals(Ki=K, Ms=Ms)

    samples_to_avg = 1000 #10000

    ''' generate voltage '''
    dev.set_vals(t_pulse = 1e-9)
    sim_weights_v = []
    for V in voltages:
        sim_weights_v.append(helper.avg_weight_across_samples(dev, V, RA, T, samples_to_avg))
    ''''''

    ''' generate pd '''
    '''
    dev.set_mag_vector()
    sim_weights_pd = []
    for t in pulse_durations:
        dev.set_vals(t_pulse = t)
        sim_weights_pd.append(helper.avg_weight_across_samples(dev, V, RA, T, samples_to_avg))
    '''
    ''''''

    error_v = np.sum( np.abs(sim_weights_v - v_weights) / v_weights )

    return error_v


if __name__ == "__main__":
    main()
