import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../fortran_source')
import numpy as np
import glob

from mtj_types_v3 import SWrite_MTJ_rng
import helper_exp_verify as helper


# assume 300K
T = 300
RA = 3.18e-12

pulse_durations = np.loadtxt('./exp_data/fig1aLR.txt',usecols=0)
print(pulse_durations)
input()
t_weights = np.loadtxt('./exp_data/fig1aLR.txt',usecols=1)
voltages = np.flip([ -v for v in np.loadtxt('./exp_data/fig1bLR300.txt',usecols=0)])
v_weights = np.loadtxt('./exp_data/fig1bLR300.txt',usecols=1)
mean_v_weight = np.mean(v_weights)
v_weights[0] = 1e-3
t_weights[0] = 1e-3

#Ms = 5.80769e5 - (2.62206e2)*(T**1.058)
#K = (2.6e-9 * 6.14314e-5)*(Ms**1.708613)
K=2.95*4.1128e-4
Ms=0.35*4.73077e5
dev = SWrite_MTJ_rng()
dev.set_vals(0)
dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha = 0.016)

def main():

    K_range = np.linspace(0.9, 1.1, 18)
    Ms_range = np.linspace(0.9, 1.1, 18)

    result = search(error_function, K_range, Ms_range)
    f = open("opt.txt", 'w')
    f.write(str(result))
    f.close()

def search(f, K_range, Ms_range):
    min_e = 1e8
    best_pair = None
    for Kf_i in K_range:
        for Mf_i in Ms_range:
            e = f(Kf_i, Mf_i)
            if e < min_e:
                min_e = e
                best_pair = (Kf_i, Mf_i)
    return (best_pair, min_e)

def error_function(K_fact, Ms_fact):
    dev.set_mag_vector()
    Ms_current = Ms_fact * Ms
    #K_theor = (2.6e-9 * 6.14314e-5)*(Ms_current**1.708613)
    dev.set_vals(Ki=K_fact*K, Ms=Ms_current)

    samples_to_avg = 250 #10000

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
    #error_pd = np.sum( np.abs(sim_weights_pd - t_weights) / t_weights )
    #error = (error_v + error_pd)/2.0
    error = error_v

    return error


if __name__ == "__main__":
    main()
