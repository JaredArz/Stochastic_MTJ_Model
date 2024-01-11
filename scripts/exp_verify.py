# Verification of MTJ Model in fortran_source against arXiv:2310.18779v2:
# Temperature-Resilient True Random Number Generation with
# Stochastic Actuated Magnetic Tunnel Junction Devices by Rehm, Morshed, et al.

# Code written by Jared Arzate, 12/29/23

# main expects file heirarchy:
#       git repo
#          |
#  --------------------------
#  |                 |      |
#  |                 |      |
#  fortran_source   dir  output dir
#                    |
#               this_script.py

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../fortran_source')

import os
import time
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.style as style
import scienceplots
plt.rc('text', usetex=True)
plt.style.use(['science'])
from scipy.signal import savgol_filter
import re
import scipy.optimize as optimize

from interface_funcs import mtj_sample
from mtj_types_v3 import SWrite_MTJ_rng
import plotting_funcs as pf

# === Constants ===
RA_product = 3.18e-12
#RA_product = 3e-12
#RA_product = 10e-12
#RA_product = 6.3e-12
# working in voltage since paper does,
# fortran code takes current so conversion is here
# NOTE: assume ohmic relationship
V_to_J = lambda V:  V/RA_product
# =================

# V_50/t_50 in this code will mean the voltage/time to get a 0.5 probability of switching

def main():
    print("===========================================================")
    print("Script takes an optional argument. Call with path to data folder if plotting existing data.")
    print("Uncomment function call within script to choose between fig 1,2,etc.")
    print("===========================================================")
    if len(sys.argv) == 1:
        start_time = time.time()
        out_path = get_out_path()
        dev = SWrite_MTJ_rng()
        dev.set_mag_vector()
        dev.set_vals(0)
        # NOTE: Ms found from Duc-The Ngo et al 2014 J. Phys. D: Appl. Phys. 47
        # NOTE: Rp is RA/A assuming RA is Rp*A

        #dev.set_vals(a=40e-9, b=40e-9, TMR = 2.03, tf = 2.6e-9, Rp = 2530, Ki=0, Ms=0)
        dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, Ki=0, Ms=0)
        #dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2230, Ki=0, Ms=0) ooops mightive been using
        #dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 5849, Ki=0, Ms=0)
        dev.set_vals(alpha=0.016)

        gen_fig1_data(dev, out_path)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif len(sys.argv) == 2:
        dir_path = sys.argv[1]
        make_and_plot_fig1(dir_path)
    else:
        print("too many arguments")

def error_function(args):
    Ms_factor, Ki_factor = args
    return a**2 + b**2 + c**2

def optimize():
    initial_guess = [0.5, 2] #ms, ki
    result = optimize.minimize(error_function, initial_guess)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)
    return

def gen_fig1_data(dev, out_path):
    #print(dev)
    #samples_to_avg = 5000
    samples_to_avg = 1000
    pulse_durations = np.linspace(0, 3.2e-9, 28) #250
    voltages = np.linspace(-1.2, 0, 28) #250
    #voltages = np.linspace(0.2, 1.5, 250) #250

    V_50 = -0.715
    generate_pulse_duration_scurve(dev, pulse_durations, V_50, 300, samples_to_avg, out_path=out_path, save_flag=True)
    np.savez(f"{out_path}/metadata_pulse_duration.npz",
             pulse_durations=pulse_durations, V_50=V_50, T=300)

    dev.set_mag_vector()

    #samples_to_avg = 5000 #8000
    samples_to_avg = 1000 #8000
    #Temps = [295, 300, 305]
    Temps = [300]
    t = 1e-9
    for T in Temps:
        generate_voltage_scurve(dev, voltages, t, T, samples_to_avg, out_path=out_path, save_flag=True)
    np.savez(f"{out_path}/metadata_voltage.npz",
             voltages=voltages, pulse_duration=t, Temps=Temps)


def make_and_plot_fig1(dir_path):
    fig_v, ax_v = pf.plot_init()
    fig_t, ax_t = pf.plot_init()
    colormap = plt.cm.get_cmap('viridis', 5)

    # voltage scurves
    metadata = np.load(glob.glob(dir_path + "/*metadata_voltage*")[0])
    pulse_duration = metadata["pulse_duration"]
    voltages = metadata["voltages"]
    pulse_amplitude = [ np.abs(v) for v in voltages ]
    for i,f in enumerate(glob.glob(dir_path + "/*voltage_sweep*")):
      print("plotting")
      f_data = np.load(f)
      weights = f_data["weights"]
      T = f_data["T"]
      ax_v.scatter(pulse_amplitude, weights, color=colormap(i), s=6)
      #ax_v.plot(pulse_amplitude, weights, color=colormap(i), label=T, alpha=0.1)
    ax_v.set_xlabel('Pulse Amplitude [v]')
    ax_v.set_ylim([0, 1])
    #ax_v.set_xlim([-0.2, 10.1])
    ax_v.set_ylabel('Weight')
    ax_v.set_title('Coin Bias')

    # pulse duration scurves
    f_data = np.load(glob.glob(dir_path + "/*pulse_duration_sweep*")[0])
    weights = f_data["weights"]
    metadata = np.load(glob.glob(dir_path + "/*metadata_pulse*")[0])
    pulse_durations = metadata["pulse_durations"]
    pulse_durations_ns = [t * 1e9 for t in pulse_durations]
    ax_t.scatter(pulse_durations_ns, weights, color=colormap(0), s=6)
    #ax_t.plot(pulse_durations_ns, weights, color=colormap(i), alpha=0.1)
    ax_t.set_xlabel('Pulse Duration [ns]')
    ax_t.set_ylabel('Weight')
    ax_t.set_ylim([0, 1])
    ax_t.set_title('Coin Bias')

    fig1aLR_x = [t*1e9 for t in (np.loadtxt('./fig1aLR.txt',usecols=0))]
    fig1aLR_y = np.loadtxt('./fig1aLR.txt',usecols=1)
    fig1bLR295_x = np.loadtxt('./fig1bLR295.txt',usecols=0)
    fig1bLR295_y = np.loadtxt('./fig1bLR295.txt',usecols=1)
    fig1bLR300_x = np.loadtxt('./fig1bLR300.txt',usecols=0)
    fig1bLR300_y = np.loadtxt('./fig1bLR300.txt',usecols=1)
    fig1bLR305_x = np.loadtxt('./fig1bLR305.txt',usecols=0)
    fig1bLR305_y = np.loadtxt('./fig1bLR305.txt',usecols=1)

    ax_t.scatter(fig1aLR_x, fig1aLR_y, color=colormap(1), s=12, marker='^', alpha = 1, label = "experiment")
    #ax_v.scatter(fig1bLR295_x, fig1bLR295_y, color=colormap(1), s=1.5, alpha = 1)
    ax_v.scatter(fig1bLR300_x, fig1bLR300_y, color=colormap(1), s=12, marker='^', alpha = 1, label = "experiment")
    #ax_v.scatter(fig1bLR305_x, fig1bLR305_y, color=colormap(1), s=1.5, alpha = 1)

    ax_v.legend(prop={'size': 12})
    ax_t.legend(prop={'size': 12})

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_t, f"../results/scurve_dataset_{date_match.group(0)}/fig1a.svg")
    pf.prompt_save_svg(fig_v, f"../results/scurve_dataset_{date_match.group(0)}/fig1b.svg")

def gen_fig2_data(dev, out_path):
    # Take voltage corresponding to 0.5 probability for pulse duration of 1ns at 300K
    # and compute the change in probability around that voltage for +-5 [K] to get a measure of dp/dT
    #
    # Then for a range of pulse durations, repeat with a new 50% voltage for each
    #
    # for the default device configuration, V_50 = 0.3940 at 1ns, 300K
    # ====

    print(dev)
    samples_to_avg = 10000 #10000
    pulse_durations = [1e-9, 2e-9, 3e-9]
    voltages = np.linspace(-0.9 ,0 , 200)

    T_delta = 20
    Temps = (300-T_delta, 300+T_delta)

    V_50s = []
    for pulse_duration in pulse_durations:
        for T in Temps:
            generate_voltage_scurve(dev, voltages, pulse_duration, T, samples_to_avg,
                                         out_path=out_path, save_flag=True)
        V_50s.append(generate_voltage_scurve(dev, voltages, pulse_duration, 300, samples_to_avg,
                                                  save_flag=False))

    np.savez(f"{out_path}/metadata_fig2.npz",
             voltages=voltages, V_50s=V_50s, Temps=Temps, pulse_durations=pulse_durations)

def make_and_plot_fig2(dir_path):
    fig, ax = pf.plot_init()
    fig_v, ax_v = pf.plot_init()

    metadata = np.load(glob.glob(dir_path + "/*metadata*")[0])
    pulse_durations = metadata["pulse_durations"]
    Temps = metadata["Temps"]
    voltages = metadata["voltages"]
    V_50s = metadata["V_50s"]

    colormap = plt.cm.get_cmap('viridis', len(pulse_durations)+1)
    files = glob.glob(dir_path + "/*voltage_sweep*")

    dT = (Temps[1] - Temps[0])
    print(f"dT: {dT}")
    dpdT = []
    #TODO: add measure of stddev
    for i, pulse_duration in enumerate(pulse_durations):
        # plot a pair of scurves for each pulse duration in addition
        # to calculating dp/dT for good measure
        V_50_idx = find_idx_at_nearest(voltages, V_50s[i])
        print(f"V50: {voltages[V_50_idx]}")
        f_data_T1 = np.load( match_file(files, pulse_duration, Temps[1], 0) )
        f_data_T0 = np.load( match_file(files, pulse_duration, Temps[0], 0) )
        weights_T1 = f_data_T1["weights"]
        weights_T0 = f_data_T0["weights"]
        weights_T1_smoothed = savgol_filter(weights_T1, 50, 9)
        weights_T0_smoothed = savgol_filter(weights_T0, 50, 9)
        #dp = weights_T1[V_50_idx] - weights_T0[V_50_idx]
        dp = weights_T1_smoothed[V_50_idx] - weights_T0_smoothed[V_50_idx]
        #print(f"dp: p_T1 - p_T0 = {weights_T1_smoothed[V_50_idx]} - {weights_T0_smoothed[V_50_idx]}" )
        print(f"--- {dp}")
        dpdT.append(dp/dT)
        ax_v.scatter(voltages, weights_T0, s=0.05, color=colormap(i))
        ax_v.scatter(voltages, weights_T1, s=0.05, color=colormap(i))
        ax_v.plot(voltages, weights_T0_smoothed, alpha = 0.5, color=colormap(i), label=Temps[0])
        ax_v.plot(voltages, weights_T1_smoothed, alpha = 0.5, color=colormap(i), label=Temps[1])

    ax_v.set_xlabel('Voltage [v]')
    ax_v.set_ylabel('Weight')
    ax_v.set_title('Coin Bias')
    ax_v.legend()

    ax.stem(pulse_durations, dpdT)
    ax.axhline(np.log(2)/(2*300))
    ax.set_xscale('log')

    ax.set_xlabel('Pulse Duration [s]')
    ax.set_ylabel('dp/dT [K-1]')
    ax.set_title('Coin Bias')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_v, f"../results/scurve_dataset_{date_match.group(0)}/fig2_curves.svg")
    pf.prompt_save_svg(fig, f"../results/scurve_dataset_{date_match.group(0)}/fig2.svg")

def gen_fig3_data(dev, out_path):
    # Generate voltage scurve and directly compute a discrete dp/dV around p=0.5.
    # Repeat for a variety of pulse durations. All at 300K
    # ====

    samples_to_avg = 5000
    pulse_durations = [1e-9, 2e-9, 3e-9, 4e-9]
    voltages = np.linspace(-0.9,0,250)

    T = 300

    V_50s = []
    for pulse_duration in pulse_durations:
        V_50s.append(generate_voltage_scurve(dev, voltages, pulse_duration, T, samples_to_avg,
                                             out_path=out_path, save_flag=True))

    np.savez(f"{out_path}/metadata_fig3.npz",
             voltages=voltages, V_50s=V_50s, pulse_durations=pulse_durations)

def make_and_plot_fig3(dir_path):
    fig_a, ax_a = pf.plot_init()
    fig_b, ax_b = pf.plot_init()

    metadata = np.load(glob.glob(dir_path + "/*metadata*")[0])
    pulse_durations = metadata["pulse_durations"]
    voltages = metadata["voltages"]
    pulse_amplitude = [ np.abs(v) for v in voltages ]
    V_50s = metadata["V_50s"]

    colormap = plt.cm.get_cmap('viridis', len(pulse_durations)+1)
    files = glob.glob(dir_path + "/*voltage_sweep*")

    dpdV = []
    #TODO: add measure of stddev
    for i, pulse_duration in enumerate(pulse_durations):
        V_50_idx = find_idx_at_nearest(voltages, V_50s[i])
        f_data = np.load(match_file(files, pulse_duration, 300, 0))
        weights = f_data["weights"]
        #weights_smoothed = savgol_filter(weights, 50, 7)
        #FIXME smooth or step
        #dp = (weights_smoothed[V_50_idx + 1]) - (weights_smoothed[V_50_idx - 1])
        dp = (weights[V_50_idx + 3]) - (weights[V_50_idx - 3])
        dV = (voltages[V_50_idx + 3]) - (voltages[V_50_idx - 3])
        dpdV.append(dp/dV)
        #Fig 3 a
        ax_a.scatter(pulse_amplitude, weights, s=0.05, color=colormap(i), alpha=0.7)
        #ax_a.plot(pulse_amplitude, weights_smoothed,label=pulse_duration, color=colormap(i))

    ax_a.set_title('Coin Bias')
    ax_a.set_xlabel('Pulse Amplitude [v]')
    ax_a.set_ylabel('p')
    ax_a.legend()

    ax_b.stem(pulse_durations, dpdV)
    #TODO add analytical curve...
    ax_b.set_xscale('log')

    ax_b.set_xlabel('Pulse Duration [s]')
    ax_b.set_ylabel('|dp/dV| [V-1]')
    ax_b.set_title('Sensitivity to Voltage Amplitude')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_a, f"../results/scurve_dataset_{date_match.group(0)}/fig3a.svg")
    pf.prompt_save_svg(fig_b, f"../results/scurve_dataset_{date_match.group(0)}/fig3b.svg")

def gen_fig4_data(dev, out_path):
    # for a range of pulse durations, calculate V 50 then generate an scurve to compute dp/dt from

    samples_to_avg = 2500
    T = 300
    pulse_durations = np.linspace(0, 20e-9, 250)
    voltages = np.linspace(-0.9, 0, 250)
    t_50s = [1e-9, 2e-9, 3e-9, 4e-9]

    V_50s = []
    for t_50 in t_50s:
        V_50 = generate_voltage_scurve(dev, voltages, t_50, T, samples_to_avg, save_flag=False)
        V_50s.append(V_50)
        generate_pulse_duration_scurve(dev, pulse_durations, V_50, T, samples_to_avg,
            out_path=out_path, save_flag=True)

    np.savez(f"{out_path}/metadata_fig4.npz",
             V_50s=V_50s, pulse_durations=pulse_durations,
             t_50s=t_50s, T=T)


def make_and_plot_fig4(dir_path):
    fig_a, ax_a = pf.plot_init()
    fig_b, ax_b = pf.plot_init()

    metadata = np.load(glob.glob(dir_path + "/*metadata*")[0])
    pulse_durations = metadata["pulse_durations"]
    pulse_durations_ns = [t * 1e9 for t in pulse_durations]
    V_50s = metadata["V_50s"]
    t_50s = metadata["t_50s"]

    colormap = plt.cm.get_cmap('viridis', len(t_50s)+1)
    files = glob.glob(dir_path + "/*pulse_duration_sweep*")

    dpdt = []
    #TODO: add measure of stddev
    for i, t_50 in enumerate(t_50s):
        t_50_idx = find_idx_at_nearest(pulse_durations, t_50)
        f_data = np.load(match_file(files, V_50s[i], 300, 0))
        weights = f_data["weights"]
        dp = (weights[t_50_idx + 3]) - (weights[t_50_idx - 3])
        dt = (pulse_durations[t_50_idx + 3]) - (pulse_durations[t_50_idx - 3])
        dpdt.append(dp/dt)
        #Fig 4 a
        ax_a.scatter(pulse_durations, weights, s=0.5, color=colormap(i))
        ax_a.plot(pulse_durations, weights, alpha=0.5, color=colormap(i), label= f"{V_50s[i]:.2e}")

    ax_a.set_title('Coin Bias')
    ax_a.set_xlabel('Pulse Durations [ns]')
    ax_a.set_ylabel('p')
    ax_a.legend()

    tdpdt = [t_i*dpdt_i for t_i, dpdt_i in zip(t_50s, dpdt)]
    ax_b.stem(t_50s, tdpdt)
    ax_b.set_xscale('log')

    ax_b.set_xlabel('t, Pulse Duration [s]')
    ax_b.set_ylabel('t*dp/dt')
    ax_b.set_title('Sensitivity to Pulse Duration')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_a, f"../results/scurve_dataset_{date_match.group(0)}/fig4a.svg")
    pf.prompt_save_svg(fig_b, f"../results/scurve_dataset_{date_match.group(0)}/fig4b.svg")
# ================================================================




# ================ helper functions for generating data =======================

def match_file(strings, x, T, i) -> str:
    file_pattern = re.compile((fr'x{x:.2e}_T{T}_i{i}.npz').replace('.',"\."))
    for s in strings:
        if file_pattern.search(s): return s

def avg_weight_across_samples(dev,V,T,samples_to_avg) -> float:
    sum_of_samples = np.sum([(mtj_sample(dev,V_to_J(V),T=T))[0] for _ in range(samples_to_avg)])
    return sum_of_samples/samples_to_avg

def find_idx_at_nearest(arr, val) -> int:
    return (np.abs(np.asarray(arr)-val)).argmin()

def generate_pulse_duration_scurve(dev, durations, V, T, samples_to_avg, i=0, out_path=None, save_flag=True) -> float:
    weights = []
    if save_flag and out_path is None:
        print("No outpath")
        exit()
    elif save_flag:
        print(f"Generating pulse duration scurve with V: {V:.2e}v, T: {T}K, iteration: {i}")
    for t in durations:
        dev.set_vals(t_pulse = t)
        weights.append(avg_weight_across_samples(dev, V, T, samples_to_avg))
    t_50 = durations[find_idx_at_nearest(weights, 0.5)]
    if save_flag:
        np.savez(f"{out_path}/pulse_duration_sweep_weights_x{V:.2e}_T{T}_i{i}.npz", weights=weights, T=T)
    return t_50

def generate_voltage_scurve(dev, voltages, t, T, samples_to_avg, i=0, out_path=None, save_flag=True) -> float:
    dev.set_vals(t_pulse = t)
    weights = []
    if save_flag and out_path is None:
        print("No outpath")
        exit()
    elif save_flag:
        print(f"Generating voltage scurve with t: {t:.2e}s, T: {T}K, iteration: {i}")
    for V in voltages:
        weights.append(avg_weight_across_samples(dev, V, T, samples_to_avg))
    V_50 = voltages[find_idx_at_nearest(weights, 0.5)]
    if save_flag:
        np.savez(f"{out_path}/voltage_sweep_scurve_data_x{t:.2e}_T{T}_i{i}.npz", weights=weights, T=T)
    return V_50

def get_out_path() -> str:
    make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
    #create dir and write path
    date = datetime.now().strftime("%H:%M:%S")
    out_path = f"../results/scurve_dataset_{date}"
    make_dir("../results")
    make_dir(f"{out_path}")
    return out_path

# ===============================================




if __name__ == "__main__":
    main()
