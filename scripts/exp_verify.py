import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../fortran_source')

# main expects file heirarchy:
#       git repo
#          |
#  --------------------------
#  |                 |      |
#  |                 |      |
#  fortran_source   dir  output dir
#                    |
#               this_script.py

import os
import time
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import re

from interface_funcs import mtj_sample
from mtj_types_v3 import SWrite_MTJ_rng
import plotting_funcs as pf

# === Constants ===
samples_to_avg = 100 #10000 is pretty smooth
RA_product = 3e-12
# working in voltage since paper does,
# fortran code takes current so conversion is here
V_to_J = lambda V:  V/RA_product
# =================



# ========== main functions =================
def gen_fig1_data():
    start_time = time.time()
    out_path = get_out_path()
    #FIXME using negative voltages currently
    #V_lut = np.linspace(0,0.9,V_steps)
    voltages = np.linspace(-0.9,0,500)
    pulse_durations = np.linspace(0,3e-9,500)
    dev = SWrite_MTJ_rng()
    #FIXME need correct params
    #dev.set_vals(a=40e-9, b=40e-9, TMR=2.03, tf=2.6e-9) #paper values
    dev.set_vals(0)
    dev.set_mag_vector()

    V_50 = -0.3940 # FIXME: paper value is 0.715v
    gen_pulse_duration_scurve(dev, pulse_durations, V_50, 300, out_path)
    np.savez(f"{out_path}/metadata_pulse_duration.npy",
             pulse_durations=pulse_durations, V_50=V_50, T=300)

    Temps = [200, 300, 400]
    pulse_duration = 1e-9
    for T in Temps:
        gen_voltage_scurve(dev, voltages, pulse_duration, T, out_path)
    np.savez(f"{out_path}/metadata_voltage.npz",
             voltages=voltages, pulse_duration=pulse_duration, Temps=Temps)

    print("--- %s seconds ---" % (time.time() - start_time))

def gen_fig2_data():
    # Take 0.5 probability voltage for pulse duration of 1ns at 300K
    #
    # Then for a range of pulse durations,
    # compute change in probability for +-5K to get dp/dT
    # repeat multiple times and take stddev since variation in probability tends to
    # increase with temperature
    #
    # for the old device configuration, v=0.3940 and ΔT should be 100 to see any effect
    # ====

    v50 = 0.3940
    start_time = time.time()
    out_path = get_out_path()
    #FIXME long times take quite a bit
    #t_lut = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    pulse_durations = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    voltages = np.linspace(v50-0.25,v50+0.25,100)
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_mag_vector()

    num_iters = 3
    Temps = [200, 400]
    for i in range(num_iters):
        for pulse_duration in pulse_durations:
            for T in Temps:
              gen_voltage_scurve(dev, voltages, pulse_durations, T, out_path):
    w_file = f"{out_path}/metadata.npz"
    np.savez(w_file, voltages=voltages, V50=v50, Temps=Temps, num_iters=num_iters, pulse_durations=pulse_durations)
    print("--- %s seconds ---" % (time.time() - start_time))

def make_fig1():
    fig_v, ax_v = pf.plot_init()
    fig_t, ax_t = pf.plot_init()
    dir_path = sys.argv[1]
    colormap = plt.cm.get_cmap('viridis', 5)

    # voltage scurves
    data_files = glob.glob(dir_path + "/*voltage_sweep*")
    metadata_file = glob.glob(dir_path + "/*metadata_voltage*")[0]
    metadata = np.load(metadata_file)
    pulse_duration = metadata["pulse_duration"]
    voltages = metadata["voltages"]
    for i,f in enumerate(data_files):
      f_data = np.load(f)
      weights = f_data["weights"]
      T = f_data["T"]
      ax_v.plot(voltages, weights, color=colormap(i), alpha=0.7, label=T)
    ax_v.legend()
    ax_v.set_xlabel('Voltage [v]')
    ax_v.set_ylabel('Weight')
    ax_v.set_title('Coin Bias')

    # pulse duration scurves
    data_file = glob.glob(dir_path + "/*pulse_duration_sweep*")[0]
    metadata_file = glob.glob(dir_path + "/*metadata_pulse*")[0]
    weights = np.load(data_file)
    metadata = np.load(metadata_file)
    pulse_durations = metadata["pulse_durations"]
    pulse_durations_ns = [t * 1e9 for t in pulse_durations]
    T = metadata["T"]
    V_50 = metadata["V_50"]
    ax_t.plot(pulse_durations_ns, weights, color=colormap(0), alpha=0.7)
    ax_t.set_xlabel('Pulse Duration [ns]')
    ax_t.set_ylabel('Weight')
    ax_t.set_title('Coin Bias')

    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    plot_path_v = f"../results/scurve_dataset_{date_match.group(0)}/voltage_scurve.svg"
    plot_path_t = f"../results/scurve_dataset_{date_match.group(0)}/pulse_duration_scurve.svg"
    pf.prompt_save_svg(fig_v,plot_path_v)
    pf.prompt_save_svg(fig_t,plot_path_t)

def make_fig2():
    fig, ax = pf.plot_init()
    dir_path = sys.argv[1]
    colormap = plt.cm.get_cmap('viridis', 1)
    plot_dpdT(ax, dir_path, colormap)
    pf.prompt_show()
    date_match = re.search(r'\d{2}:\d{2}:\d{2}', dir_path)
    plot_path = f"../results/scurve_dataset_{date_match.group(0)}/fig2.svg"
    pf.prompt_save_svg(fig,plot_path)

# ================================================================






# ================ to generate data =======================

def gen_pulse_duration_scurve(dev, durations, V, T, out_path, i=0):
  weights = []
  print(f"Generating pulse duration scurve with V: {V}v, T: {T}K, iteration: {i}")
  for t in durations:
    dev.set_vals(t_pulse = t)
    weights.append(avg_sample(dev, V, T))
  np.savez(f"{out_path}/pulse_duration_sweep_weights_v{V}_T{T}_i{i}.npy", weights=weights, T=T)
  return

def gen_voltage_scurve(dev, voltages, t, T, out_path, i=0):
  dev.set_vals(t_pulse = t)
  weights = []
  print(f"Generating voltage scurve with t: {t}s, T: {T}K")
  for V in voltages:
    weights.append(avg_sample(dev, V, T))
  np.savez(f"{out_path}/voltage_sweep_scurve_data_T{T}_i{i}.npy", weights=weights, T=T)
  return

def avg_sample(dev,V,T):
  wght_sum = 0.0
  for _ in range(samples_to_avg): wght_sum += (mtj_sample(dev,V_to_J(V),T=T))[0]
  return wght_sum/samples_to_avg

def get_out_path():
  make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
  #create dir and write path
  date = datetime.now().strftime("%H:%M:%S")
  out_path = (f"../results/scurve_dataset_{date}")
  make_dir("../results")
  make_dir(f"{out_path}")
  return out_path

# ===============================================








# =================== to plot data ======================


def plot_dpdT(out_path, colormap):
      color = colormap(0)
      find_idx_of_nearest = lambda arr, val: (np.abs(np.asarray(arr)-val)).argmin()
      metadata_file = glob.glob(out_path + "/*metadata*")[0]
      num_iters = metadata_file["num_iters"]
      voltages = metadata_file["voltages"]
      V50 = metadata_file["V50"]
      pulse_durations = metadata_file["pulse_durations"]
      Temps = metadata_file["Temps"]
      V50_idx = find_idx_of_nearest(voltages, value=V50)

      dT = Temps[1] - Temps[0]
      avg_dpdT = []
      current_dpdT = 0.0
      for pulse_duration in pulse_durations:
          for i in range(num_iters):
              ps = []
              for T in Temps:
                  f = glob.glob(out_path + "/*pulse_duration_sweep*T{T}*i{i}*")
                  ps.append((f["weights"])[v50_idx])
              current_dpdT += ps[1] - p[0]
            avg_dpdT.append(current_dpdT/num_iters)

      ax.plot(pulse_durations, avg_dpdT, color=color, alpha=0.7)

      ax.set_xlabel('Pulse Duration [s]')
      ax.set_ylabel('dp/dT [K-1]')
      ax.set_title('Coin Bias')

def plot_dpdV(out_path, color):
      pass

# ===============================================





if __name__ == "__main__":
  # call script with path to data folder as first opt arg to plot
  if len(sys.argv) == 2:
    print("===========================================================")
    print("script argument should be path to data folder for plotting.")
    print("uncomment function call within script to choose fig 1,2,etc.")
    print("===========================================================")

    #make_fig1()
    #make_fig2()
  else:
    gen_fig2_data()
