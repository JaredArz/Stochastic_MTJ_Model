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
import re
from interface_funcs import mtj_sample
from mtj_types_v3 import SWrite_MTJ_rng
import plotting_funcs as pf

# main expects file heirarchy:
#      repo
#        |
#      -----
#      |   |
#      |   |
#     dir  |--output_dir
#      |
# this_script.py

# === Constants ===
samples_to_avg = 250
RA_product = 3e-12
V_to_J = lambda V:  V/RA_product
# =================

def main_gen():
  start_time = time.time()
  out_path = get_out_path()
  # working in voltage since paper does,
  # fortran code takes current so conversion is here
  V_steps = 100
  #V_lut = np.linspace(0,0.9,V_steps)
  V_lut = np.linspace(-0.9,0,V_steps)
  t_steps = 100
  t_lut = np.linspace(0,3e-9,t_steps)
  dev = SWrite_MTJ_rng()
  #dev.set_vals(a=40e-9, b=40e-9, TMR=2.03, tf=2.6e-9) #paper values
  dev.set_vals(0)
  dev.set_mag_vector()
  gen_pulse_duration_scurve(dev, t_lut, -0.715, 300, out_path)
  gen_voltage_scurve(dev, V_lut, 1e-9, 300, out_path)
  print("--- %s seconds ---" % (time.time() - start_time))

def main_plot():
    colormap = plt.cm.get_cmap('viridis', 3)
    plot_voltage(sys.argv[1], colormap(0))

def gen_pulse_duration_scurve(dev, durations, V, T, out_path):
  weights = []
  for t in durations:
    dev.set_vals(t_pulse = t)
    print(f"Current pulse duration: {t}s")
    weights.append(sample(dev, V, T))
  w_file = f"{out_path}/pulse_duration_sweep_scurve_data.npz"
  np.savez(w_file, V=V, T=T, pulse_durations=durations, weights=weights)

def gen_voltage_scurve(dev, voltages, t, T, out_path):
  dev.set_vals(t_pulse = t)
  weights = []
  for V in voltages:
    print(f"Current voltage: {V}v")
    weights.append(sample(dev, V, T))
  w_file = f"{out_path}/voltage_sweep_scurve_data.npz"
  np.savez(w_file, t=t, T=T, voltages=voltages, weights=weights)

def sample(dev,V,T):
    avg_wght = 0.0
    for _ in range(samples_to_avg):
      out, _ = mtj_sample(dev,V_to_J(V),T=T)
      avg_wght = avg_wght + out
    avg_wght = avg_wght/samples_to_avg
    return avg_wght

def get_out_path():
  make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
  #create dir and write path
  date = datetime.now().strftime("%H:%M:%S")
  out_path = (f"../results/scurve_dataset_{date}")
  make_dir("../results")
  make_dir(f"{out_path}")
  return out_path

def plot_voltage(out_path, color):
  fig, ax = pf.plot_init()
  path_to_file = os.path.join(out_path, "voltage_sweep_scurve_data.npz")

  f_data = np.load(path_to_file)
  pulse_duration = f_data["t"]
  T = f_data["T"]
  voltages = f_data["voltages"]
  weights = f_data["weights"]

  ax.plot(voltages, weights, color=color, alpha=0.7)

  ax.set_xlabel('Voltage [v]')
  ax.set_ylabel('Weight')
  ax.set_title('Coin Bias')

  pf.prompt_show()
  match = re.search(r'\d{2}:\d{2}:\d{2}', path_to_file)
  dir_path = f"./results/scurve_dataset_{match.group(0)}/voltage_scurve.png"
  pf.prompt_save_svg(fig,dir_path)

def plot_voltage(out_path, color):
  fig, ax = pf.plot_init()
  path_to_file = os.path.join(out_path, "pulse_duration_sweep_scurve_data.npz")

  f_data = np.load(path_to_file)
  pulse_durations = f_data["pulse_durations"]
  T = f_data["T"]
  V = f_data["V"]
  weights = f_data["weights"]
  pulse_durations_ns = [t * 1e9 for t in pulse_durations]

  ax.plot(pulse_durations_ns, weights, color=color, alpha=0.7)

  ax.set_xlabel('Pulse Duration [ns]')
  ax.set_ylabel('Weight')
  ax.set_title('Coin Bias')

  pf.prompt_show()
  match = re.search(r'\d{2}:\d{2}:\d{2}', path_to_file)
  dir_path = f"./results/scurve_dataset_{match.group(0)}/pulse_duration_scurve.png"
  pf.prompt_save_svg(fig,dir_path)

if __name__ == "__main__":
  # call script with path to data folder as first opt arg to plot
  if len(sys.argv) == 2:
    main_plot()
  else:
    main_gen()
