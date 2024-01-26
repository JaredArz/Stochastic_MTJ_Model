import sys
sys.path.append("../")
import helper_funcs as helper

import os
import re
import numpy as np
from datetime import datetime

# ================ helper functions for generating data =======================

def generate_pulse_duration_scurve(dev, durations, V, T, samples_to_avg, i=0, out_path=None, save_flag=True) -> float:
    weights = []
    if save_flag and out_path is None:
        print("No outpath")
        exit()
    elif save_flag:
        print(f"Generating pulse duration scurve with V: {V:.2e}v, T: {T}K, iteration: {i}")
    for t in durations:
        dev.set_vals(t_pulse = t)
        weights.append(helper.avg_weight_across_samples(dev, V, samples_to_avg))
    t_50 = durations[helper.find_idx_at_nearest(weights, 0.5)]
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
        weights.append(helper.avg_weight_across_samples(dev, V, samples_to_avg))
    V_50 = voltages[helper.find_idx_at_nearest(weights, 0.5)]
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

def match_file(strings, x, T, i) -> str:
    file_pattern = re.compile((fr'x{x:.2e}_T{T}_i{i}.npz').replace('.',"\."))
    for s in strings:
        if file_pattern.search(s): return s

# ===============================================
