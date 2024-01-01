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
import re

from interface_funcs import mtj_sample
from mtj_types_v3 import SWrite_MTJ_rng
import plotting_funcs as pf

# === Constants ===
#FIXME: paper value, not default MTJ
RA_product = 3e-12
# working in voltage since paper does,
# fortran code takes current so conversion is here
#FIXME: assume ohmic relationship
V_to_J = lambda V:  V/RA_product
# =================



# ========== main functions =================
def gen_fig1_data():
    start_time = time.time()
    samples_to_avg = 1000 #10000 is smooth
    out_path = get_out_path()
    #FIXME using negative voltages currently, positives do not work
    voltages = np.linspace(-0.9, 0, 500)
    pulse_durations = np.linspace(0, 3e-9, 500)
    dev = SWrite_MTJ_rng()
    #FIXME need correct params
    #dev.set_vals(a=40e-9, b=40e-9, TMR=2.03, tf=2.6e-9) #paper values
    dev.set_vals(0)
    dev.set_mag_vector()

    V_50 = generate_voltage_scurve(dev, voltages, 1e-9, 300, samples_to_avg, save_flag=False)
    generate_pulse_duration_scurve(dev, pulse_durations, V_50, 300, samples_to_avg, out_path=out_path, save_flag=True)
    np.savez(f"{out_path}/metadata_pulse_duration.npz",
             pulse_durations=pulse_durations, V_50=V_50, T=300)

    samples_to_avg = 8000
    Temps = [290, 300, 310]
    pulse_duration = 1e-9
    for T in Temps:
        generate_voltage_scurve(dev, voltages, pulse_duration, T, samples_to_avg, out_path=out_path, save_flag=True)
    np.savez(f"{out_path}/metadata_voltage.npz",
             voltages=voltages, pulse_duration=pulse_duration, Temps=Temps)

    print("--- %s seconds ---" % (time.time() - start_time))

def make_and_plot_fig1():
    fig_v, ax_v = pf.plot_init()
    fig_t, ax_t = pf.plot_init()
    dir_path = sys.argv[1]
    colormap = plt.cm.get_cmap('viridis', 5)

    # voltage scurves
    metadata = np.load(glob.glob(dir_path + "/*metadata_voltage*")[0])
    pulse_duration = metadata["pulse_duration"]
    voltages = metadata["voltages"]
    for i,f in enumerate(glob.glob(dir_path + "/*voltage_sweep*")):
      f_data = np.load(f)
      weights = f_data["weights"]
      T = f_data["T"]
      ax_v.plot(voltages, weights, color=colormap(i), alpha=0.7, label=T)
    ax_v.legend()
    ax_v.set_xlabel('Voltage [v]')
    ax_v.set_ylabel('Weight')
    ax_v.set_title('Coin Bias')

    # pulse duration scurves
    f_data = np.load(glob.glob(dir_path + "/*pulse_duration_sweep*")[0])
    weights = f_data["weights"]
    metadata = np.load(glob.glob(dir_path + "/*metadata_pulse*")[0])
    pulse_durations = metadata["pulse_durations"]
    pulse_durations_ns = [t * 1e9 for t in pulse_durations]
    ax_t.plot(pulse_durations_ns, weights, color=colormap(0), alpha=0.7)
    ax_t.set_xlabel('Pulse Duration [ns]')
    ax_t.set_ylabel('Weight')
    ax_t.set_title('Coin Bias')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_v,f"../results/scurve_dataset_{date_match.group(0)}/voltage_scurve.svg")
    pf.prompt_save_svg(fig_t,f"../results/scurve_dataset_{date_match.group(0)}/pulse_duration_scurve.svg")

def gen_fig2_data():
    # Take voltage corresponding 0.5 probability for pulse duration of 1ns at 300K
    # and compute the change in probability around that voltage for +-5k to get a measure of dp/dT
    #
    # Then for a range of pulse durations, repeat with a new 50% voltage for each
    #
    # for the default device configuration, V_50 = 0.3940 at 1ns, 300K
    # ====

    start_time = time.time()
    samples_to_avg = 100 # will analyze deviation a bit differently in this script #TODO
    out_path = get_out_path()
    pulse_durations = [8e-10, 1e-9, 2e-9]
    voltages = np.linspace(-0.9,0,500)
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_mag_vector()

    num_iters = 5
    T_delta = 5
    Temps = (300-T_delta,300,300+T_delta) # two to calculate dT and 300K to calculate 0.5 probability voltage

    V_50s = []
    for pulse_duration in pulse_durations:
        for i in range(num_iters):
            for T in Temps:
                V_50 = generate_voltage_scurve(dev, voltages, pulse_duration, T, samples_to_avg,
                                               i=i, out_path=out_path, save_flag=True)
                if T == 300: V_50s.append(V_50)

    print(V_50s)
    np.savez(f"{out_path}/metadata_fig2.npz",
             voltages=voltages, V_50s=V_50s, Temps=Temps, num_iters=num_iters, pulse_durations=pulse_durations)
    print("--- %s seconds ---" % (time.time() - start_time))

def gen_fig3_data():
    # Generate voltage scurve and directly compute a discrete dp/dV around p=0.5.
    # Repeat for a variety of pulse durations. All at 300K
    # ====

    start_time = time.time()
    samples_to_avg = 100 # will analyze deviation a bit differently in this script #TODO
    out_path = get_out_path()
    pulse_durations = [1e-9, 1e-8, 1e-7]
    voltages = np.linspace(-0.9,0,250)
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_mag_vector()

    num_iters = 5
    T = 300

    #FIXME
    V_50s = []
    for pulse_duration in pulse_durations:
        for i in range(num_iters):
            V_50s.append(generate_voltage_scurve(dev, voltages, pulse_duration, T, samples_to_avg,
                                                 i=i, out_path=out_path, save_flag=True))

    np.savez(f"{out_path}/metadata_fig3.npz",
             voltages=voltages, V_50s=V_50s, num_iters=num_iters, pulse_durations=pulse_durations)
    print("--- %s seconds ---" % (time.time() - start_time))

'''
def gen_fig4_data():
    # Generate pulse duration scurve and directly compute a discrete dp/dt around p=0.5.
    # Repeat for a variety of voltage amplitudes. All at 300K
    # ====

    start_time = time.time()
    samples_to_avg = 100 # will analyze deviation a bit differently in this script #TODO
    out_path = get_out_path()
    pulse_durations_scurve = np.linspace(0, 5e-9, 100)
    pulse_durations_x = [1e-9, 5e-9, 1e-8]
    #FIXME
    voltages = [-0.35, -0.3940, -0.44]
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_mag_vector()

    num_iters = 1
    T = 300

    for V_50 in voltages:
        for i in range(num_iters):
            generate_pulse_duration_scurve(dev, pulse_durations_scurve, V_50, T, samples_to_avg,
                i=i, out_path=out_path, save_flag=True)

    np.savez(f"{out_path}/metadata_fig4.npz",
             V_50s=V_50s, num_iters=num_iters, pulse_durations_scurve=pulse_durations_scurve,
             pulse_durations_x=pulse_durations_x,T=T)
    print("--- %s seconds ---" % (time.time() - start_time))
'''



def make_and_plot_fig2():
    fig, ax = pf.plot_init()
    fig_v, ax_v = pf.plot_init()
    dir_path = sys.argv[1]

    metadata = np.load(glob.glob(dir_path + "/*metadata*")[0])
    num_iters = metadata["num_iters"]
    pulse_durations = metadata["pulse_durations"]
    Temps = metadata["Temps"]
    voltages = metadata["voltages"]
    V_50s = metadata["V_50s"]
    print(V_50s)

    colormap = plt.cm.get_cmap('viridis', len(pulse_durations)+1)
    files = glob.glob(dir_path + "/*voltage_sweep*")

    dT = Temps[-1] - Temps[0]
    dpdT = []
    #TODO: add measure of stddev
    for i, pulse_duration in enumerate(pulse_durations):
        dp_sum = 0.0
        # plot a pair of scurves for each pulse duration in addition
        # to calculating dp/dT for good measure
        avg_weights_T0 = np.zeros(len(voltages))
        avg_weights_T1 = np.zeros(len(voltages))
        for j in range(num_iters):
            V_50_idx = find_idx_of_nearest(voltages, V_50s[i*num_iters + j])
            dp = 0.0
            for sign, T in zip((-1,1), (Temps[0], Temps[-1])):
                f_data = np.load( match_file("voltage", files, pulse_duration, T, j) )
                p = (f_data["weights"])[V_50_idx]
                dp += sign*p
                # scurves, unecessary
                if sign < 0:
                    current = avg_weights_T0
                else:
                    current = avg_weights_T1
                for l,w in enumerate(f_data["weights"]):
                    current[l] += w
            dp_sum += dp
        dpdT.append(dp_sum/(dT*num_iters))
        ax_v.plot(voltages, [avg/num_iters for avg in avg_weights_T0], color=colormap(i), alpha=0.7)
        ax_v.plot(voltages, [avg/num_iters for avg in avg_weights_T1], color=colormap(i), alpha=0.7)

    ax.stem(pulse_durations, dpdT)
    ax.axhline(0.0016)
    ax.set_xscale('log')

    ax.set_xlabel('Pulse Duration [s]')
    ax.set_ylabel('dp/dT [K-1]')
    ax.set_title('Coin Bias')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_v, f"../results/scurve_dataset_{date_match.group(0)}/fig2_curves.svg")
    pf.prompt_save_svg(fig, f"../results/scurve_dataset_{date_match.group(0)}/fig2.svg")

def make_and_plot_fig3():
    fig_a, ax_a = pf.plot_init()
    fig_b, ax_b = pf.plot_init()
    dir_path = sys.argv[1]

    metadata = np.load(glob.glob(dir_path + "/*metadata*")[0])
    num_iters = metadata["num_iters"]
    pulse_durations = metadata["pulse_durations"]
    voltages = metadata["voltages"]
    V_50s = metadata["V_50s"]

    colormap = plt.cm.get_cmap('viridis', len(pulse_durations)+1)
    files = glob.glob(dir_path + "/*voltage_sweep*")

    dpdV = []
    #TODO: add measure of stddev
    for i, pulse_duration in enumerate(pulse_durations):
        dp_sum = 0.0
        avg_weights = np.zeros(len(voltages))
        for j in range(num_iters):
            #FIXME
            V_50_idx = find_idx_of_nearest(voltages, V_50s[i*num_iters+j])
            f_data = np.load( match_file("voltage", files, pulse_duration, 300, j) )
            weights = f_data["weights"]
            dp_sum += (weights[V_50_idx + 1]) - (weights[V_50_idx - 1])
            for k, weight in enumerate(f_data["weights"]): avg_weights[k] += weight
        dV = (voltages[V_50_idx + 1]) - (voltages[V_50_idx - 1])
        #FIXME: flipping sign to account for negative voltage
        dpdV.append(-dp_sum/(dV*num_iters))
        #Fig 3 a
        ax_a.plot(voltages, [avg/num_iters for avg in avg_weights], color=colormap(i), alpha=0.7)

    ax_a.set_title('Coin Bias')
    ax_a.set_xlabel('Voltage [v]')
    ax_a.set_ylabel('p')

    ax_b.stem(pulse_durations, dpdV)
    #TODO add analytical curve...
    ax_b.set_xscale('log')

    ax_b.set_xlabel('Pulse Duration [s]')
    ax_b.set_ylabel('dp/dV [V-1]')
    ax_b.set_title('Sensitivity to Voltage Amplitude')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_a, f"../results/scurve_dataset_{date_match.group(0)}/fig3a.svg")
    pf.prompt_save_svg(fig_b, f"../results/scurve_dataset_{date_match.group(0)}/fig3b.svg")

'''
def make_and_plot_fig4():
    fig_a, ax_a = pf.plot_init()
    fig_b, ax_b = pf.plot_init()
    dir_path = sys.argv[1]

    metadata = np.load(glob.glob(dir_path + "/*metadata*")[0])
    num_iters = metadata["num_iters"]
    pulse_durations = metadata["pulse_durations"]
    pulse_durations_ns = [t * 1e9 for t in pulse_durations]
    voltages = metadata["voltages"]

    colormap = plt.cm.get_cmap('viridis', len(voltages)+1)
    files = glob.glob(dir_path + "/*pulse_duration_sweep*")

    dpdt = []
    #TODO: add measure of stddev
    for i, pulse_duration in enumerate(pulse_durations):
        dp_sum = 0.0
        avg_weights = np.zeros(len(pulse_durations))
        for j in range(num_iters):
            #FIXME
            f_data = np.load( match_file("pulse", files, voltages[i], 300, j) )
            weights = f_data["weights"]
            t_50_idx = find_idx_of_nearest(weights, 0.5) # parallel arrays
            dp_sum += (weights[t_50_idx + 1]) - (weights[t_50_idx - 1])
            for k, weight in enumerate(weights): avg_weights[k] += weight
        dt = (pulse_durations[t_50_idx + 1]) - (pulse_durations[t_50_idx - 1])
        dpdt.append(-dp_sum/(dt*num_iters))
        #Fig 4 a
        ax_a.plot(pulse_durations_ns, [avg/num_iters for avg in avg_weights], color=colormap(i), alpha=0.7)

    ax_a.set_title('Coin Bias')
    ax_a.set_xlabel('Pulse Durations [ns]')
    ax_a.set_ylabel('p')

    tdpdt = [t_i*dpdt_i for t_i, dpdt_i in zip(pulse_durations, dpdt)]
    print(tdpdt)
    ax_b.stem(pulse_durations, tdpdt)
    ax_b.set_xscale('log')

    ax_b.set_xlabel('Pulse Duration [s]')
    ax_b.set_ylabel('t*dp/dt')
    ax_b.set_title('Sensitivity to Pulse Duration')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_a, f"../results/scurve_dataset_{date_match.group(0)}/fig4a.svg")
    pf.prompt_save_svg(fig_b, f"../results/scurve_dataset_{date_match.group(0)}/fig4b.svg")
'''
# ================================================================






# ================ helper functions for generating data =======================

def match_file(str_option, strings, t, T, i) -> str:
    if str_option == "voltage":
        file_pattern = re.compile((fr't{t:.2e}_T{T}_i{i}.npz').replace('.',"\."))
    elif str_option == "pulse":
        file_pattern = re.compile((fr'v{v}_T{T}_i{i}.npz').replace('.',"\."))
    for s in strings:
        if file_pattern.search(s): return s

def avg_weight_across_samples(dev,V,T,samples_to_avg) -> float:
    sum_of_samples = np.sum([(mtj_sample(dev,V_to_J(V),T=T))[0] for _ in range(samples_to_avg)])
    return sum_of_samples/samples_to_avg

def find_idx_of_nearest(arr, val) -> int:
    return (np.abs(np.asarray(arr)-val)).argmin()

def generate_pulse_duration_scurve(dev, durations, V, T, samples_to_avg, i=0, out_path=None, save_flag=True) -> float:
    weights = []
    if save_flag and out_path is None:
        print("No outpath")
        exit()
    elif save_flag:
        print(f"Generating pulse duration scurve with V: {V}v, T: {T}K, iteration: {i}")
    for t in durations:
        dev.set_vals(t_pulse = t)
        weights.append(avg_weight_across_samples(dev, V, T, samples_to_avg))
    t_50 = pulse_durations[find_idx_of_nearest(weights, 0.5)]
    if save_flag:
        np.savez(f"{out_path}/pulse_duration_sweep_weights_v{V}_T{T}_i{i}.npz", weights=weights, T=T)
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
    V_50 = voltages[find_idx_of_nearest(weights, 0.5)]
    if save_flag:
        np.savez(f"{out_path}/voltage_sweep_scurve_data_t{t:.2e}_T{T}_i{i}.npz", weights=weights, T=T)
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
    print("===========================================================")
    print("Script takes an optional argument. Call with path to data folder if plotting existing data.")
    print("Uncomment function call within script to choose between fig 1,2,etc.")
    print("===========================================================")
    if len(sys.argv) == 2:

        make_and_plot_fig1()
        #make_and_plot_fig2()
        #make_and_plot_fig4()
        #make_and_plot_fig3()
    else:
        #gen_fig3_data()
        #gen_fig4_data()
        #gen_fig2_data()
        gen_fig1_data()
