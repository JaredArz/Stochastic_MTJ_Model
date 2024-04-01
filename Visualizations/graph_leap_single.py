import os
import sys
import glob
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from paretoset import paretoset
from scipy import stats
from scipy.special import rel_entr
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel("ERROR")

sys.path.append("../")
from Testing.prng_dist import prng_dist


SOT_color = "#3f8efc"
SOT_color2 = "#87bfff"
STT_color = "#613dc1"
STT_color2 = "#aeb8fe"
pareto_color ="#d8315b"
target_pdf_color = "#8b8c89"
prng_pdf_color = "#07070a"

param_ranges = {
  "alpha"   : (0.01, 0.1),
  "Ki"      : (0.2e-3, 1e-3),
  "Ms"      : (0.3e6, 2e6),
  "K_295"   : (0.2e-3, 1e-3),
  "Ms_295"  : (0.3e6, 2e6),
  "Rp"      : (500, 50000),
  "eta"     : (0.1, 2),
  "J_she"   : (0.01e12, 5e12),
  "t_pulse" : (0.5e-9, 75e-9),
  "t_relax" : (0.5e-9, 75e-9)
}

def get_df(device, pdf_type):
  if device == "SOT":
    df = pd.DataFrame(columns=["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax", "kl_div_score", "energy", "xxis", "countData"])
    param_path = f"../SOT_RNG_Leap/SOT_{pdf_type}/parameters"
  elif device == "STT":
    df = pd.DataFrame(columns=["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax", "kl_div_score", "energy", "xxis", "countData"])
    param_path = f"../STT_RNG_Leap/STT_{pdf_type}/parameters"
  else:
    raise ValueError("Incorrect device type")

  for i, pklFile in enumerate(glob.glob(os.path.join(param_path, "*.pkl"))):
    with open(pklFile, "rb") as f:
      data = pickle.load(f)
      
      if device == "SOT":
        alpha = data["genome"][0]
        Ki = data["genome"][1]
        Ms = data["genome"][2]
        Rp = data["genome"][3]
        eta = data["genome"][4]
        J_she = data["genome"][5]
        t_pulse = data["genome"][6]
        t_relax = data["genome"][6]
        kl_div_score = data["kl_div"]
        energy = data["energy"]
        xxis = data["xxis"]
        countData = data["countData"]
        row = [alpha, Ki, Ms, Rp, eta, J_she, t_pulse, t_relax, kl_div_score, energy, xxis, countData]
      
      elif device == "STT":
        alpha = data["genome"][0]
        K_295 = data["genome"][1]
        Ms_295 = data["genome"][2]
        Rp = data["genome"][3]
        t_pulse = data["genome"][4]
        t_relax = data["genome"][4]
        kl_div_score = data["kl_div"]
        energy = data["energy"]
        xxis = data["xxis"]
        countData = data["countData"]
        row = [alpha, K_295, Ms_295, Rp, t_pulse, t_relax, kl_div_score, energy, xxis, countData]
      
      df.loc[len(df)] = row

  df = df.sort_values(by="kl_div_score")
  df.reset_index(drop=True, inplace=True)

  return df


def top_distributions(device, pdf_type, top=5):
  if device == "SOT":
    best_color = SOT_color
    avg_color = SOT_color2
  elif device == "STT":
    best_color = STT_color
    avg_color = STT_color2
  else:
    raise ValueError("Incorrect device type")

  xxis, target_pdf, prng_pdf = prng_dist(samples=100_000)
  df = get_df(device, pdf_type)
  df_top = df.head(top)

  pdf_arr = []
  for i, row in df_top.iterrows():
    pdf_arr.append(row["countData"])

  pdf_mean = np.average(pdf_arr, axis=0)
  pdf_std = np.std(pdf_arr, axis=0)

  fig, axs = plt.subplots(layout="constrained")
  linewidth = 3
  tick_size = 28
  title_size = 34
  axis_size = 30
  legend_size = 28

  axs.plot(xxis, pdf_mean, color=avg_color, linewidth=linewidth, label="Top 5 Avg.")
  axs.fill_between(xxis, pdf_mean-pdf_std, pdf_mean+pdf_std, alpha=0.5, facecolor=avg_color, edgecolor=avg_color)
  axs.plot(xxis, pdf_arr[0], color=best_color, linewidth=linewidth, label="Best")
  axs.plot(xxis, prng_pdf, color=prng_pdf_color, linewidth=linewidth, label="PRNG")
  axs.plot(xxis, target_pdf, color=target_pdf_color, linewidth=linewidth, linestyle="dashed", label="Target")
  axs.tick_params(axis="both", which="major", labelsize=tick_size)
  axs.set_title(f"{device} PDF Comparison (LEAP)", size=title_size, weight="bold")
  axs.set_xlabel("Generated Number", size=axis_size, weight="bold")
  axs.set_ylabel("Probability", size=axis_size, weight="bold")
  axs.legend(fontsize=legend_size)
  plt.show()


def get_norm(range):
  norm = matplotlib.colors.Normalize(vmin=range[0], vmax=range[1])
  return norm


def parameter_heatmap(device, pdf_type, top=5):
  if device == "SOT":
    params = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax"]
    ylabels = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax"]
  elif device == "STT":
    params = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax"]
    ylabels = ["alpha", "Ki", "Ms", "Rp", "t_pulse", "t_relax"]
  else:
    raise ValueError("Incorrect device type")
  
  df = get_df(device, pdf_type)
  df_top = df.head(top)
  xlabels = [f"config_{i}" for i in range(len(df_top))]

  data = []
  for i, row in df_top.iterrows():
    data.append(row[params].to_dict())
  
  fig, axs = plt.subplots(layout="constrained")
  marker = "s"
  marker_size = 2500
  w = 0.8
  h = 1
  swatch_size = 24
  rotation = 0
  linewidth = 1
  tick_size = 28
  title_size = 34
  axis_size = 30
  color = "coolwarm"
  cmap = matplotlib.cm.get_cmap(color)

  axs.set_xticks(np.arange(0,len(xlabels),1))
  axs.set_yticks(np.arange(0,len(ylabels),1))
  axs.set_xticklabels(xlabels)
  axs.set_yticklabels(ylabels)

  for i, dict in enumerate(data):
    for j, param in enumerate(dict.keys()):
      param_val = dict[param]
      cmap_val = get_norm(param_ranges[param])(param_val)
      axs.scatter(i, j, s=marker_size, marker=marker, c=cmap(cmap_val))
      axs.add_patch(Rectangle(xy=(i-w/2, j-h/2), width=w, height=h, linewidth=linewidth, edgecolor="black", facecolor=cmap(cmap_val)))
      axs.text(i, j, f"{param_val:.1e}", size=swatch_size, weight="bold", rotation=rotation, horizontalalignment="center", verticalalignment="center",
              color = "black" if (cmap_val > 0.25 and cmap_val < 0.75) else "white")

  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  cbar = fig.colorbar(sm)
  cbar.ax.yaxis.set_ticks([0,1])
  cbar.ax.set_yticklabels(["Min", "Max"], fontsize=tick_size)
  axs.tick_params(axis="both", which="major", labelsize=tick_size)
  axs.set_title(f"{device} Top Parameter Configurations (LEAP)", size=title_size, weight="bold")
  axs.set_xlabel("Config ID", size=axis_size, weight="bold")
  axs.set_ylabel("Parameters", size=axis_size, weight="bold")
  plt.show()


def get_probe_df(device, pdf_type):
  if device == "SOT":
    probe_df = pd.DataFrame(columns=["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax", "kl_div_score", "energy"])
    probe_path = f"../SOT_RNG_Leap/SOT_{pdf_type}/probe_output"
  elif device == "STT":
    probe_df = pd.DataFrame(columns=["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax", "kl_div_score", "energy"])
    probe_path = f"../STT_RNG_Leap/STT_{pdf_type}/probe_output"
  else:
    raise ValueError("Incorrect device type")

  for _, csvFile in enumerate(glob.glob(os.path.join(probe_path, "*.csv"))):
    df = pd.read_csv(csvFile)
    for _, row in df.iterrows():
      fitness = ast.literal_eval(row["fitness"])
      genome = ast.literal_eval(row["genome"])
      genome.append(genome[-1])
      probe_df.loc[len(probe_df)] = genome + fitness

  return probe_df


def get_dist(df, param, bins, samples):
  counts, _ = np.histogram(df[param].to_list(), bins=bins)
  counts = counts/samples
  xxis = np.linspace(param_ranges[param][0], param_ranges[param][1], bins)

  return xxis, counts


def parameter_exploration(device, pdf_type):
  if device == "SOT":
    color = SOT_color
    rows = 2
    cols = 4
    params = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax"]
  elif device == "STT":
    color = STT_color
    rows = 2
    cols = 3
    params = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax"]
  else:
    raise ValueError("Incorrect device type")
  
  probe_df = get_probe_df(device, pdf_type)
  bins = 15
  samples = len(probe_df)
  
  param_dict = dict()
  for param in params:
    xxis, counts = get_dist(probe_df, param, bins, samples)
    param_dict[f"{param}_xxis"] = xxis
    param_dict[f"{param}_counts"] = counts
  
  fig, axs = plt.subplots(rows, cols, sharey=True, layout="constrained")
  tick_size = 28
  subtitle_size = 34
  title_size = 34
  axis_size = 30

  for row in range(rows):
    for col in range(cols):
      param = params[row*cols + col]
      xxis = param_dict[f"{param}_xxis"]
      counts = param_dict[f"{param}_counts"]

      axs[row,col].plot(xxis, counts, color=color, linewidth=3)
      axs[row,col].set_xticks([param_ranges[param][0], param_ranges[param][1]], visible=True, rotation="horizontal")
      axs[row,col].tick_params(axis="both", which="major", labelsize=tick_size)
      axs[row,col].xaxis.get_offset_text().set_fontsize(tick_size)
      param = "Ki" if param == "K_295" else param
      param = "Ms" if param == "Ms_295" else param
      axs[row,col].set_title(param, size=subtitle_size)

  fig.suptitle(f"{device} Parameter Exploration (LEAP)", size=title_size, weight="bold")
  fig.supxlabel("Parameter Range", size=axis_size, weight="bold")
  fig.supylabel("Probability", size=axis_size, weight="bold")
  plt.show()


def pareto_front(device, pdf_type):
  if device == "SOT":
    color = SOT_color
  elif device == "STT":
    color = STT_color
  else:
    raise ValueError("Incorrect device type")

  probe_df = get_probe_df(device, pdf_type)

  df = probe_df[ (probe_df["kl_div_score"] != 1000000) & (probe_df["energy"] != 1000000) ]
  pareto_df = df[["kl_div_score", "energy"]]
  mask = paretoset(pareto_df, sense=["min", "min"])
  pareto_df = df[mask]

  fig, axs = plt.subplots(layout="constrained")
  marker_size = 300
  tick_size = 28
  title_size = 34
  axis_size = 30
  legend_size = 28

  axs.scatter(df["kl_div_score"], df["energy"], color=color, s=marker_size, label="Samples")
  axs.scatter(pareto_df["kl_div_score"], pareto_df["energy"], color=pareto_color, s=marker_size, label="Pareto Front")
  axs.tick_params(axis="both", which="major", labelsize=tick_size)
  axs.yaxis.get_offset_text().set_fontsize(tick_size)
  axs.set_title(f"{device} Pareto Front (LEAP)", size=title_size, weight="bold")
  axs.set_xlabel("KL Divergence Score", size=axis_size, weight="bold")
  axs.set_ylabel("Energy", size=axis_size, weight="bold")
  axs.legend(fontsize=legend_size)
  plt.show()



if __name__ == "__main__":
  device = "SOT"
  # device = "STT"
  pdf_type = "gamma"

  # top_distributions(device, pdf_type, top=5)
  parameter_heatmap(device, pdf_type, top=5)
  # parameter_exploration(device, pdf_type)
  # pareto_front(device, pdf_type)