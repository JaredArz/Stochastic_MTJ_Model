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
t_weights = np.loadtxt('./exp_data/fig1aLR.txt',usecols=1)
voltages = np.flip([ -v for v in np.loadtxt('./exp_data/fig1bLR300.txt',usecols=0)])
v_weights = np.loadtxt('./exp_data/fig1bLR300.txt',usecols=1)
mean_v_weight = np.mean(v_weights)
v_weights[0] = 1e-3

Ms = 5.80769e5 - (2.62206e2)*(T**1.058)
K = (2.6e-9 * 6.14314e-5)*(Ms**1.708613)
dev = SWrite_MTJ_rng()
dev.set_vals(0)
dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530)

def main():
    '''
    print(f"pulse dur {pulse_durations}")
    print(f"volta {voltages}")
    print(f"vwe {v_weights}")
    print(f"twe {t_weights}")
    '''

    K_range = np.linspace(1.4, 1.7, 50)
    Ms_range = np.linspace(0.25, 0.5, 50)
    #K_range = [1.5388]
    #Ms_range = [0.35134]


    result = search(error_function, K_range, Ms_range)
    print(result)

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
    #print( (K_fact*K, Ms_current) )

    samples_to_avg = 250 #10000

    ''' generate voltage '''
    dev.set_vals(t_pulse = 1e-9)
    sim_weights = []
    for V in voltages:
        sim_weights.append(helper.avg_weight_across_samples(dev, V, RA, T, samples_to_avg))
    ''''''

    #dev.set_mag_vector()
    #helper.generate_pulse_duration_scurve(dev,pulse_durations,-0.715,RA,300,samples_to_avg,out_path=out_path,save_flag=True)
    # get data for pulse duration here too
    # XXX

    e = np.sum( (sim_weights - v_weights) / v_weights )

    return e


if __name__ == "__main__":
    main()
