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
import matplotlib.pyplot as plt
import matplotlib.style as style
import scienceplots
plt.rc('text', usetex=True)
plt.style.use(['science'])
from scipy.signal import savgol_filter
import re
import matplotlib.lines as mlines

from mtj_types_v3 import SWrite_MTJ_rng
import plotting_funcs as pf
import helper_exp_verify as helper

# === Constants ===
RA = 3.18e-12
# working in voltage since paper does,
# fortran code takes current so conversion is here
# NOTE: assumes ohmic relationship
# =================

# V_50/t_50 in this code will mean the voltage/time to get a 0.5 probability of switching

def get_mk(T):
    Tc = 1453
    n = 1.804
    q = 1.0583
    Kstar = 4.389e5
    Mstar = 5.8077e5
    Ms_295 = 165576.94999
    K_295 = 0.001161866/(2.6e-9)
    cm = Ms_295 - Mstar*( 1 - (295/Tc)**q )
    ck = K_295 - Kstar*( ( Ms_295/Mstar )**n )
    #cm = 0
    #ck = 0

    Ms = Mstar*( 1 - (T/Tc)**q ) + cm
    K = 2.6e-9*((Kstar)*( (Ms/Mstar)**n ) + ck)

    return (K,Ms)

def main():
    print("===========================================================")
    print("Script takes an optional argument. Call with path to data folder if plotting existing data.")
    print("Uncomment function call within script to choose between fig 1,2,etc.")
    print("===========================================================")
    if len(sys.argv) == 1:
        start_time = time.time()
        out_path = helper.get_out_path()
        print(out_path)
        dev = SWrite_MTJ_rng()
        dev.set_mag_vector()
        dev.set_vals(0)
        # NOTE: Ms found from Duc-The Ngo et al 2014 J. Phys. D: Appl. Phys. 47
        # NOTE: Rp is RA/A assuming RA is Rp*A

        #Ms = 5.80769e5 - (2.62206e2)*(300**1.058) + 946270.189018873
        #K = 2.6e-9 * ((6.14314e-5)*(Ms**1.708613) + 141623.699473255)
        #Keff = 2.6e-9 * ( (K/2.6e-9) - (2*np.pi*10**(-7) * Ms**2) )
        #dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha=0.016, Ki=1.46735*K, Ms=0.3673*Ms)
        dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, alpha=0.016)
        #dev.set_vals(Ki=2.85*4.1128e-4, Ms=0.35*4.73077e5)
        #dev.set_vals(Ki=2.825*4.1128e-4, Ms=0.35*4.73077e5)
        #dev.set_vals(Ms=473076.923077, Ki = 158192.307692 * 2.6e-9)
        #dev.set_vals(Ki=2.75*4.1128e-4, Ms=0.4*4.73077e5)
        #dev.set_vals(Ki=0.0012132710579346336, Ms=165573.63124478742)
        #dev.set_vals(Ms = 165576.94999999998, Ki = 0.0011618660000000001)
        K, Ms = get_mk(300)
        print(K,Ms)
        dev.set_vals(Ki=K, Ms=Ms)
        print(dev)

        gen_fig3_data(dev, out_path)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif len(sys.argv) == 2:
        dir_path = sys.argv[1]
        make_and_plot_fig3(dir_path)
    else:
        print("too many arguments")


def gen_fig1_data(dev, out_path):
    #samples_to_avg = 10000
    samples_to_avg = 10000
    pulse_durations = [3.0e-10, 4.0e-10, 5.0e-10, 6.0e-10, 7.0e-10, 8.0e-10, 9.0e-10, 1.0e-09, 1.1e-09,
               1.2e-09, 1.3e-09, 1.4e-09, 1.5e-09, 1.6e-09, 1.7e-09, 1.8e-09, 1.9e-09, 2.0e-09,
               2.1e-09, 2.2e-09, 2.3e-09, 2.4e-09, 2.5e-09, 2.6e-09, 2.7e-09, 3.0e-09, 3.5e-09,
               4.0e-09]
    voltages = np.linspace(-0.97919268, -0.43084478, 29) #250

    V_50 = -0.715
    helper.generate_pulse_duration_scurve(dev, pulse_durations, V_50, RA, 300, samples_to_avg, out_path=out_path, save_flag=True)
    np.savez(f"{out_path}/metadata_pulse_duration.npz",
             pulse_durations=pulse_durations, V_50=V_50, T=300)

    dev.set_mag_vector()

    #samples_to_avg = 10000 #8000
    samples_to_avg = 11000
    Temps = [295, 300, 305]
    t = 1e-9
    for T in Temps:
        K, Ms = get_mk(T)
        dev.set_vals(Ki=K, Ms=Ms)
        helper.generate_voltage_scurve(dev, voltages, RA, t, T, samples_to_avg, out_path=out_path, save_flag=True)
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
    colors = ['k','r','b']
    for i,f in enumerate(glob.glob(dir_path + "/*voltage_sweep*")):
      print("plotting")
      f_data = np.load(f)
      weights = f_data["weights"]
      T = f_data["T"]
      ax_v.scatter(pulse_amplitude, weights,color=colors[i],marker='^', s=15)
      ax_v.plot(pulse_amplitude, weights, linestyle = 'dashed', color=colors[i])
      #ax_v.plot(pulse_amplitude, weights, color=colormap(i), label=T, alpha=0.1)
    ax_v.set_xlabel('Pulse Amplitude [v]')
    ax_v.set_ylim([0, 1])
    #ax_v.set_xlim([-0.2, 10.1])
    ax_v.set_ylabel('p')
    ax_v.set_title('Coin Bias')

    # pulse duration scurves
    f_data = np.load(glob.glob(dir_path + "/*pulse_duration_sweep*")[0])
    weights = f_data["weights"]
    metadata = np.load(glob.glob(dir_path + "/*metadata_pulse*")[0])
    pulse_durations = metadata["pulse_durations"]
    pulse_durations_ns = [t * 1e9 for t in pulse_durations]
    ax_t.scatter(pulse_durations_ns, weights, color='r' , marker='^', s=12)
    ax_t.plot(pulse_durations_ns, weights, 'r', linestyle='dashed' )
    #ax_t.plot(pulse_durations_ns, weights, color=colormap(i), alpha=0.1)
    ax_t.set_xlabel('Pulse Duration [ns]')
    ax_t.set_ylabel('p')
    ax_t.set_ylim([0, 1])
    ax_t.set_title('Coin Bias')

    fig1aLR_x = [t*1e9 for t in (np.loadtxt('./exp_data/fig1aLR.txt',usecols=0))]
    fig1aLR_y = np.loadtxt('./exp_data/fig1aLR.txt',usecols=1)
    fig1bLR295_x = np.loadtxt('./exp_data/fig1bLR295.txt',usecols=0)
    fig1bLR295_y = np.loadtxt('./exp_data/fig1bLR295.txt',usecols=1)
    fig1bLR300_x = np.loadtxt('./exp_data/fig1bLR300.txt',usecols=0)
    fig1bLR300_y = np.loadtxt('./exp_data/fig1bLR300.txt',usecols=1)
    fig1bLR305_x = np.loadtxt('./exp_data/fig1bLR305.txt',usecols=0)
    fig1bLR305_y = np.loadtxt('./exp_data/fig1bLR305.txt',usecols=1)

    ax_t.plot(fig1aLR_x, fig1aLR_y, color='b', alpha=0.33)
    ax_t.scatter(fig1aLR_x, fig1aLR_y, color='b', marker='s', s=12,alpha=0.5)
    ax_v.plot(fig1bLR295_x, fig1bLR295_y, 'm', label = "295k", alpha=0.25)
    ax_v.plot(fig1bLR300_x, fig1bLR300_y, 'k', label = "300K", alpha=0.25)
    ax_v.plot(fig1bLR305_x, fig1bLR305_y, 'c', label = "305K",alpha=0.25)
    ax_v.scatter(fig1bLR295_x, fig1bLR295_y, color='m', marker='s', s=12,alpha=0.2)
    ax_v.scatter(fig1bLR300_x, fig1bLR300_y, color='k', marker='s', s=12,alpha=0.2)
    ax_v.scatter(fig1bLR305_x, fig1bLR305_y, color='c', marker='s', s=12,alpha=0.2)

    line1 = mlines.Line2D([], [], color=colormap(1), marker='s',
                          markersize=6, label='experiment')
    line2 = mlines.Line2D([], [], color=colormap(1), marker='^',
                          markersize=6, label='macrospin model')
    line3 = mlines.Line2D([], [], color=colormap(1), marker='s',
                          markersize=6, label='experiment')
    line4 = mlines.Line2D([], [], color=colormap(1), marker='^',
                          markersize=6, label='macrospin model')
    ax_v.legend(handles=[line1, line2])
    ax_t.legend(handles=[line3, line4])
    #ax_v.legend(prop={'size': 12})
    #ax_t.legend(prop={'size': 12})

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_t, f"../results/scurve_dataset_{date_match.group(0)}/fig1a.svg")
    pf.prompt_save_svg(fig_v, f"../results/scurve_dataset_{date_match.group(0)}/fig1b.svg")
# ================================================================







# ================================================================
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
    pulse_durations = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    voltages = np.linspace(-0.97919268, -0.43084478, 100) #250

    T_delta = 5
    Temps = (300-T_delta, 300+T_delta)

    V_50s = []
    for pulse_duration in pulse_durations:
        for T in Temps:
            K, Ms = get_mk(T)
            dev.set_vals(Ki=K, Ms=Ms)
            helper.generate_voltage_scurve(dev, voltages,RA, pulse_duration, T, samples_to_avg,
                                         out_path=out_path, save_flag=True)
        V_50s.append(helper.generate_voltage_scurve(dev, voltages,RA, pulse_duration, 300, samples_to_avg,
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
        V_50_idx = helper.find_idx_at_nearest(voltages, V_50s[i])
        print(f"V50: {voltages[V_50_idx]}")
        f_data_T1 = np.load( helper.match_file(files, pulse_duration, Temps[1], 0) )
        f_data_T0 = np.load( helper.match_file(files, pulse_duration, Temps[0], 0) )
        weights_T1 = f_data_T1["weights"]
        weights_T0 = f_data_T0["weights"]
        dp = weights_T1[V_50_idx] - weights_T0[V_50_idx]
        #dp = weights_T1_smoothed[V_50_idx] - weights_T0_smoothed[V_50_idx]
        #print(f"dp: p_T1 - p_T0 = {weights_T1_smoothed[V_50_idx]} - {weights_T0_smoothed[V_50_idx]}" )
        print(f"--- {dp}")
        dpdT.append(dp/dT)
        ax_v.scatter(voltages, weights_T0, s=5, color=colormap(i))
        ax_v.scatter(voltages, weights_T1, s=5, color=colormap(i))
        #ax_v.plot(voltages, weights_T0_smoothed, alpha = 0.5, color=colormap(i), label=Temps[0])
        #ax_v.plot(voltages, weights_T1_smoothed, alpha = 0.5, color=colormap(i), label=Temps[1])

    ax_v.set_xlabel('Voltage [v]')
    ax_v.set_ylabel('Weight')
    ax_v.set_title('Coin Bias')

    ax.scatter(pulse_durations, dpdT, color = 'r', s=15, marker='^')
    ax.axhline(np.log(2)/(2*300), linestyle = 'dashed')
    ax.set_xscale('log')

    ax.set_xlabel('Pulse Duration [s]')
    ax.set_ylabel('dp/dT [K-1]')
    ax.set_title('Sensitivity to Pulse Duration')

    # experimental results
    fig2LR_exp_x = np.loadtxt('./exp_data/fig2LR_exp.txt',usecols=0)
    fig2LR_exp_y = np.loadtxt('./exp_data/fig2LR_exp.txt',usecols=1)
    fig2LR_FP_x = np.loadtxt('./exp_data/fig2LR_FP.txt',usecols=0)
    fig2LR_FP_y = np.loadtxt('./exp_data/fig2LR_FP.txt',usecols=1)

    ax.scatter(fig2LR_exp_x, fig2LR_exp_y, color='k', s=12, marker='s', alpha = 1)
    ax.plot(fig2LR_FP_x, fig2LR_FP_y, color='0.7', alpha = 1)

    line1 = mlines.Line2D([], [], color='k', marker='s',
                          markersize=6, label='experiment')
    line2 = mlines.Line2D([], [], color='r', marker='^',
                          markersize=6, label='macrospin model')
    line3 = mlines.Line2D([], [], color='0.7', linestyle='-',
                          markersize=6, label='Fokker-Planck')
    line4 = mlines.Line2D([], [], color='b', linestyle='--',
                          markersize=6, label='Short-pulse limit')
    ax.legend(handles=[line1, line2, line3, line4])
    #ax.legend(prop={'size': 12})

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_v, f"../results/scurve_dataset_{date_match.group(0)}/fig2_curves.svg")
    pf.prompt_save_svg(fig, f"../results/scurve_dataset_{date_match.group(0)}/fig2.svg")
# ================================================================









# ================================================================
def gen_fig3_data(dev, out_path):
    # Generate voltage scurve and directly compute a discrete dp/dV around p=0.5.
    # Repeat for a variety of pulse durations. All at 300K
    # ====

    samples_to_avg = 10000
    pulse_durations = [1e-11, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    voltages = np.linspace(-0.97919268, -0.43084478, 150) #250

    T = 300

    V_50s = []
    for pulse_duration in pulse_durations:
        V_50s.append(helper.generate_voltage_scurve(dev, voltages,RA, pulse_duration, T, samples_to_avg,
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
        V_50_idx = helper.find_idx_at_nearest(voltages, V_50s[i])
        f_data = np.load(helper.match_file(files, pulse_duration, 300, 0))
        weights = f_data["weights"]
        #weights_smoothed = savgol_filter(weights, 50, 7)
        #FIXME smooth or step
        #dp = (weights_smoothed[V_50_idx + 1]) - (weights_smoothed[V_50_idx - 1])
        dp = (weights[V_50_idx + 1]) - (weights[V_50_idx - 1])
        dV = (-voltages[V_50_idx + 1]) - (-voltages[V_50_idx - 1])
        dpdV.append(dp/dV)
        #Fig 3 a
        ax_a.scatter(pulse_amplitude, weights, s=5, color=colormap(i), alpha=1, label = pulse_duration)
        #ax_a.plot(pulse_amplitude, weights_smoothed,label=pulse_duration, color=colormap(i))

    ax_a.set_title('Coin Bias')
    ax_a.set_xlabel('Pulse Amplitude [v]')
    ax_a.set_ylabel('p')

    ax_b.scatter(pulse_durations[1:], dpdV[1:], color='r', marker='^', s = 15)
    ax_b.plot(pulse_durations[1:], dpdV[1:], linestyle='dashed', color='r')
    #TODO add analytical curve...
    ax_b.set_xscale('log')

    ax_b.set_xlabel('Pulse Duration [s]')
    ax_b.set_ylabel('dp/dV [V-1]')
    ax_b.set_title('Sensitivity to Voltage Amplitude')

    # experimental results
    fig3bLR_exp_x = np.loadtxt('./exp_data/fig3bLR_exp.txt',usecols=0)
    fig3bLR_exp_y = np.loadtxt('./exp_data/fig3bLR_exp.txt',usecols=1)
    fig3bLR_FP_x = np.loadtxt('./exp_data/fig3bLR_FP.txt',usecols=0)
    fig3bLR_FP_y = np.loadtxt('./exp_data/fig3bLR_FP.txt',usecols=1)

    ax_b.scatter(fig3bLR_exp_x, fig3bLR_exp_y, color='k', s=12, marker='s', alpha = 0.9)
    ax_b.plot(fig3bLR_exp_x, fig3bLR_exp_y, color='k', alpha = 0.9)

    ax_b.plot(fig3bLR_FP_x, fig3bLR_FP_y, color='b', alpha = 1, linestyle='--')

    line1 = mlines.Line2D([], [], color='k', marker='s',
                          markersize=6, label='experiment')
    line2 = mlines.Line2D([], [], color='r', marker='^',
                          markersize=6, label='macrospin model')
    line3 = mlines.Line2D([], [], color='b', linestyle='--',
                          markersize=6, label='Short-pulse limit')
    ax_b.legend(handles=[line1, line2, line3])

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_a, f"../results/scurve_dataset_{date_match.group(0)}/fig3a.svg")
    pf.prompt_save_svg(fig_b, f"../results/scurve_dataset_{date_match.group(0)}/fig3b.svg")
# ================================================================







# ================================================================
def gen_fig4_data(dev, out_path):
    # for a range of pulse durations, calculate V 50 then generate an scurve to compute dp/dt from

    samples_to_avg = 10000
    T = 300
    pulse_durations = np.linspace(0, 50e-9, 100)
    voltages = np.linspace(-0.9, 0, 100)
    t_50s = [1e-9, 5e-9, 1e-8, 5e-8]

    V_50s = []
    for t_50 in t_50s:
        V_50 = helper.generate_voltage_scurve(dev, voltages,RA, t_50, T, samples_to_avg, save_flag=False)
        V_50s.append(V_50)
        helper.generate_pulse_duration_scurve(dev, pulse_durations, V_50,RA, T, samples_to_avg,
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
        t_50_idx = helper.find_idx_at_nearest(pulse_durations, t_50)
        f_data = np.load(match_file(files, V_50s[i], 300, 0))
        weights = f_data["weights"]
        if t_50_idx == 99:
            t_50_idx = 98
        dp = (weights[t_50_idx + 1]) - (weights[t_50_idx - 1])
        dt = (pulse_durations[t_50_idx + 1]) - (pulse_durations[t_50_idx - 1])
        dpdt.append(dp/dt)
        #Fig 4 a
        ax_a.scatter(pulse_durations, weights, s=0.5, color=colormap(i))
        ax_a.plot(pulse_durations, weights, alpha=0.5, color=colormap(i), label= f"{V_50s[i]:.2e}")

    ax_a.set_title('Coin Bias')
    ax_a.set_xlabel('Pulse Durations [ns]')
    ax_a.set_ylabel('p')

    tdpdt = [t_i*dpdt_i for t_i, dpdt_i in zip(t_50s, dpdt)]
    ax_b.stem(t_50s, tdpdt)
    ax_b.set_xscale('log')

    ax_b.set_xlabel('t, Pulse Duration [s]')
    ax_b.set_ylabel('t*dp/dt')
    ax_b.set_title('Sensitivity to Pulse Duration')

    # experimental results
    fig4bLR_exp_x = np.loadtxt('./exp_data/fig4bLR_exp.txt',usecols=0)
    fig4bLR_exp_y = np.loadtxt('./exp_data/fig4bLR_exp.txt',usecols=1)
    fig4bLR_FP_x = np.loadtxt('./exp_data/fig4bLR_FP.txt',usecols=0)
    fig4bLR_FP_y = np.loadtxt('./exp_data/fig4bLR_FP.txt',usecols=1)

    ax_b.scatter(fig4bLR_exp_x, fig4bLR_exp_y, color=colormap(1), s=12, marker='^', alpha = 1, label = "experiment")
    ax_b.plot(fig4bLR_FP_x, fig4bLR_FP_y, color=colormap(1), alpha = 1, label = "FP")

    ax_a.legend(prop={'size': 12})
    ax_b.legend(prop={'size': 12})

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    pf.prompt_save_svg(fig_a, f"../results/scurve_dataset_{date_match.group(0)}/fig4a.svg")
    pf.prompt_save_svg(fig_b, f"../results/scurve_dataset_{date_match.group(0)}/fig4b.svg")
# ================================================================





if __name__ == "__main__":
    main()
