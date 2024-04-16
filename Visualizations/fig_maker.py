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
# STT_color = "#613dc1"
# STT_color2 = "#aeb8fe"
STT_color = "#aa3e98"
STT_color2 = "#c19ee0"
# pareto_color = "#d8315b"
pareto_color = "#ff3c38"
target_pdf_color = "#8b8c89"
prng_pdf_color = "#07070a"
metric_color = "#ffee32"
null_color = "#495057"

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

def get_best_df(opt, device, pdf_type, csv_name=None):
  if device == "SOT":
    columns = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax", "kl_div_score", "energy", "xxis", "countData"]
    df = pd.DataFrame(columns=columns)
  elif device == "STT":
    columns = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax", "kl_div_score", "energy", "xxis", "countData"]
    df = pd.DataFrame(columns=columns)
  
  param_path = f"../{device}_RNG_{opt}/{device}_{pdf_type}/parameters"

  for i, pklFile in enumerate(glob.glob(os.path.join(param_path, "*.pkl"))):
    with open(pklFile, "rb") as f:
      data = pickle.load(f)
      
      if opt == "Leap":
        genome = list(data["genome"])
        genome.append(genome[-1])
        row = [data[key] for key in ["kl_div", "energy", "xxis", "countData"]]
        row = genome + row
      elif opt == "RL":
        row = [data[key] for key in columns]
      
      df.loc[len(df)] = row

  df = df.sort_values(by="kl_div_score")
  df.reset_index(drop=True, inplace=True)

  if csv_name != None:
    df = df.drop(columns=["xxis", "countData"])
    df.to_csv(csv_name, index=False)

  return df


def top_distributions(opt, device, pdf_type, top=5):
  if opt != "Leap" and opt != "RL":
    raise ValueError("Incorrect optimization type")
  
  if device == "SOT":
    best_color = SOT_color
    avg_color = SOT_color2
  elif device == "STT":
    best_color = STT_color
    avg_color = STT_color2
  else:
    raise ValueError("Incorrect device type")

  xxis, target_pdf, prng_pdf = prng_dist(samples=100_000)
  df = get_best_df(opt, device, pdf_type)
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

  alg = "EA" if opt == "Leap" else "RL"

  axs.plot(xxis, pdf_mean, color=avg_color, linewidth=linewidth, label="Top 5 Avg.")
  axs.fill_between(xxis, pdf_mean-pdf_std, pdf_mean+pdf_std, alpha=0.5, facecolor=avg_color, edgecolor=avg_color)
  axs.plot(xxis, pdf_arr[0], color=best_color, linewidth=linewidth, label="Best")
  axs.plot(xxis, prng_pdf, color=prng_pdf_color, linewidth=linewidth, label="PRNG")
  axs.plot(xxis, target_pdf, color=target_pdf_color, linewidth=linewidth, linestyle="dashed", label="Target")
  axs.tick_params(axis="both", which="major", labelsize=tick_size)
  axs.set_title(f"{device} PDF Comparison ({alg})", size=title_size, weight="bold")
  axs.set_xlabel("Generated Number", size=axis_size, weight="bold")
  axs.set_ylabel("Probability", size=axis_size, weight="bold")
  axs.legend(fontsize=legend_size)
  plt.show()


def get_norm(range):
  norm = matplotlib.colors.Normalize(vmin=range[0], vmax=range[1])
  return norm


def get_metric_cbar(top):
  opts = ["Leap", "RL"]
  devices = ["SOT", "STT"]
  pdf_types = ["gamma"]

  kl_div_arr = []
  energy_arr = []
  for opt in opts:
    for device in devices:
      for pdf_type in pdf_types:
        df = get_best_df(opt, device, pdf_type)
        kl_div_arr.append(df["kl_div_score"].head(top).to_list())
        energy_arr.append(df["energy"].head(top).to_list())
  
  kl_div_arr = list(np.asarray(kl_div_arr).flatten())
  kl_div_arr.append(0.0013569866155363818)
  kl_div_arr.append(0.013248617137107576)
  kl_div_arr.append(0.10049436637969326)
  kl_div_min = np.min(kl_div_arr)
  kl_div_max = np.max(kl_div_arr)
  kl_div_range = (kl_div_min, kl_div_max)
  kl_div_norm = get_norm(kl_div_range)

  energy_arr = list(np.asarray(energy_arr).flatten())
  energy_arr.append(1.5007049583868615e-13)
  energy_arr.append(2.754649094361683e-09)
  energy_min = np.min(energy_arr)
  energy_max = np.max(energy_arr)
  energy_range = (energy_min, energy_max)
  energy_norm = get_norm(energy_range)

  cmap = matplotlib.cm.get_cmap("summer_r")
  return cmap, kl_div_norm, energy_norm


def parameter_heatmap(opt, device, pdf_type, top=5):
  if opt != "Leap" and opt != "RL":
    raise ValueError("Incorrect optimization type")
  
  if device == "SOT":
    params = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax", "kl_div_score", "energy"]
    ylabels = ["α", r"$K_{i}$", r"$M_{s}$", r"$R_{p}$", "η", r"$J_{SOT}$", r"$t_{pulse}$", r"$t_{relax}$", "KL Div.", "Energy(J)"]
    prng = {"alpha": "--", "Ki": "--", "Ms": "--", "Rp": "--", "eta": "--", "J_she": "--", "t_pulse": "--", "t_relax": "--", "kl_div_score": 0.0013569866155363818, "energy": "--"}
    defaults = {"alpha": 0.03, "Ki": 1.0056364e-3, "Ms": 1.2e6, "Rp": 5e3, "eta": 0.3, "J_she": 5e11, "t_pulse": 10e-9, "t_relax": 15e-9, "kl_div_score": 0.013248617137107576, "energy": 1.5007049583868615e-13}
  elif device == "STT":
    params = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax","kl_div_score", "energy"]
    ylabels = ["α", r"$K_{i}$", r"$M_{s}$", r"$R_{p}$", r"$t_{pulse}$", r"$t_{relax}$", "KL Div.", "Energy(J)"]
    prng = {"alpha": "--", "K_295": "--", "Ms_295": "--", "Rp": "--", "t_pulse": "--", "t_relax": "--", "kl_div_score": 0.0013569866155363818, "energy": "--"}
    defaults = {"alpha": 0.03, "K_295": 1.0056364e-3, "Ms_295": 1.2e6, "Rp": 5e3, "t_pulse": 1e-9, "t_relax": 10e-9, "kl_div_score": 0.10049436637969326, "energy": 2.754649094361683e-09}
  else:
    raise ValueError("Incorrect device type")
  
  df = get_best_df(opt, device, pdf_type)
  df_top = df.head(top)
  xlabels = [f"{i+1}" for i in range(len(df_top))]
  xlabels.append("Default")
  xlabels.append("PRNG")

  data = []
  for i, row in df_top.iterrows():
    data.append(row[params].to_dict())
  data.append(defaults)
  data.append(prng)
  
  fig, axs = plt.subplots(layout="constrained")
  marker = "s"
  marker_size = 1000
  w = 0.95
  h = 1
  swatch_size = 20
  rotation = 0
  linewidth = 1
  tick_size = 26
  title_size = 34
  axis_size = 30
  cbar_tick_size = 20
  cmap = matplotlib.cm.get_cmap("coolwarm")
  metric_cmap, kl_div_norm, energy_norm = get_metric_cbar(top)

  axs.set_xticks(np.arange(0,len(xlabels),1))
  axs.set_yticks(np.arange(0,len(ylabels),1))
  axs.set_xticklabels(xlabels)
  axs.set_yticklabels(ylabels)
  # axs.axis["left"].major_ticklabels.set_ha("center")

  for i, dict in enumerate(data):
    for j, param in enumerate(dict.keys()):
      param_val = dict[param]

      text = param_val if param_val == "--" else f"{param_val:.1e}"

      if param == "kl_div_score":
        cmap_val = kl_div_norm(param_val)
        axs.scatter(i, j, s=marker_size, marker=marker, c=metric_cmap(cmap_val))
        axs.add_patch(Rectangle(xy=(i-w/2, j-h/2), width=w, height=h, linewidth=linewidth, edgecolor="black", facecolor=metric_cmap(cmap_val)))
        axs.text(i, j, text, size=swatch_size, weight="bold", rotation=rotation, horizontalalignment="center", verticalalignment="center", color="black")
      elif param == "energy":
        if param_val == "--":
          c = null_color
          text_color = "white"
        else:
          cmap_val = energy_norm(param_val)
          c = metric_cmap(cmap_val)
          text_color = "black"
        axs.scatter(i, j, s=marker_size, marker=marker, c=c)
        axs.add_patch(Rectangle(xy=(i-w/2, j-h/2), width=w, height=h, linewidth=linewidth, edgecolor="black", facecolor=c))
        axs.text(i, j, text, size=swatch_size, weight="bold", rotation=rotation, horizontalalignment="center", verticalalignment="center", color=text_color)
      elif xlabels[i] == "PRNG":
        axs.scatter(i, j, s=marker_size, marker=marker, c=null_color)
        axs.add_patch(Rectangle(xy=(i-w/2, j-h/2), width=w, height=h, linewidth=linewidth, edgecolor="black", facecolor=null_color))
        axs.text(i, j, text, size=swatch_size, weight="bold", rotation=rotation, horizontalalignment="center", verticalalignment="center", color="white")
      else:
        cmap_val = get_norm(param_ranges[param])(param_val)
        axs.scatter(i, j, s=marker_size, marker=marker, c=cmap(cmap_val))
        axs.add_patch(Rectangle(xy=(i-w/2, j-h/2), width=w, height=h, linewidth=linewidth, edgecolor="black", facecolor=cmap(cmap_val)))
        axs.text(i, j, text, size=swatch_size, weight="bold", rotation=rotation, horizontalalignment="center", verticalalignment="center",
                color = "black" if (cmap_val > 0.25 and cmap_val < 0.75) else "white")

  alg = "EA" if opt == "Leap" else "RL"
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  
  metric_cmap = matplotlib.cm.get_cmap("summer")
  sm = plt.cm.ScalarMappable(cmap=metric_cmap, norm=norm)
  cbar = fig.colorbar(sm, ticks=[], shrink=0.9, pad=0.01)
  cbar.set_label(label="Metric Value", size=cbar_tick_size, weight="bold")
  cbar.ax.text(0.5, -0.01, "Worst", size=cbar_tick_size, transform=cbar.ax.transAxes, va="top", ha="center")
  cbar.ax.text(0.5, 1.0, "Best", size=cbar_tick_size, transform=cbar.ax.transAxes, va="bottom", ha="center")

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  cbar = fig.colorbar(sm, ticks=[], shrink=0.9, pad=0.01)
  cbar.set_label(label="Parameter Ranges", size=cbar_tick_size, weight="bold")
  cbar.ax.text(0.5, -0.01, "Min", size=cbar_tick_size, transform=cbar.ax.transAxes, va="top", ha="center")
  cbar.ax.text(0.5, 1.0, "Max", size=cbar_tick_size, transform=cbar.ax.transAxes, va="bottom", ha="center")
  
  axs.tick_params(axis="both", which="major", labelsize=tick_size)
  axs.set_title(f"{device} Top Parameter Configurations ({alg})", size=title_size, weight="bold")
  axs.set_xlabel("Config ID", size=axis_size, weight="bold")
  axs.set_ylabel("Parameters", size=axis_size, weight="bold")
  plt.show()


def get_full_df(opt, device, pdf_type):
  if opt == "Leap":
    if device == "SOT":
      columns = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax", "kl_div_score", "energy"]
    elif device == "STT":
      columns=["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax", "kl_div_score", "energy"]
    
    full_df = pd.DataFrame(columns=columns)
    probe_path = f"../{device}_RNG_{opt}/{device}_{pdf_type}/probe_output"

    for _, csvFile in enumerate(glob.glob(os.path.join(probe_path, "*.csv"))):
      df = pd.read_csv(csvFile)
      for _, row in df.iterrows():
        fitness = ast.literal_eval(row["fitness"])
        genome = ast.literal_eval(row["genome"])
        genome.append(genome[-1])
        full_df.loc[len(full_df)] = genome + fitness
  
  elif opt == "RL":
    if device == "SOT":
      subset = ["alpha", "Ki", "Ms", "Rp", "TMR", "eta", "J_she", "t_pulse", "t_relax", "d", "tf"]
    elif device == "STT":
      subset = ["alpha", "K_295", "Ms_295", "Rp", "TMR", "t_pulse", "t_relax", "d", "tf"]
      
    csvFile = f"../{device}_RNG_RL/{device}_Gamma_Model-timestep-6000_Results.csv"
    full_df = pd.read_csv(csvFile)
    full_df = full_df.sort_values(by="kl_div_score")
    full_df.drop_duplicates(subset=subset, keep="first")
    full_df.reset_index(drop=True, inplace=True)

  return full_df



def get_dist(df, param, bins, samples):
  counts, _ = np.histogram(df[param].to_list(), bins=bins)
  counts = counts/samples
  xxis = np.linspace(param_ranges[param][0], param_ranges[param][1], bins)

  return xxis, counts


def parameter_exploration(opt, device, pdf_type):
  if opt != "Leap" and opt != "RL":
    raise ValueError("Incorrect optimization type")
  
  if device == "SOT":
    color = SOT_color
    rows = 2
    cols = 4
    params = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax"]
    titles = ["α", r"$K_{i}$", r"$M_{s}$", r"$R_{p}$", "η", r"$J_{SOT}$", r"$t_{pulse}$", r"$t_{relax}$"]
  elif device == "STT":
    color = STT_color
    rows = 2
    cols = 3
    params = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax"]
    titles = ["α", r"$K_{i}$", r"$M_{s}$", r"$R_{p}$", r"$t_{pulse}$", r"$t_{relax}$"]
  else:
    raise ValueError("Incorrect device type")
  
  full_df = get_full_df(opt, device, pdf_type)
  bins = 15
  samples = len(full_df)
  
  param_dict = dict()
  for param in params:
    xxis, counts = get_dist(full_df, param, bins, samples)
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
      title = titles[row*cols + col]
      xxis = param_dict[f"{param}_xxis"]
      counts = param_dict[f"{param}_counts"]

      axs[row,col].plot(xxis, counts, color=color, linewidth=3)
      axs[row,col].set_xticks([param_ranges[param][0], param_ranges[param][1]], visible=True, rotation="horizontal")
      axs[row,col].tick_params(axis="both", which="major", labelsize=tick_size)
      axs[row,col].xaxis.get_offset_text().set_fontsize(tick_size)
      axs[row,col].set_title(title, size=subtitle_size)

  alg = "EA" if opt == "Leap" else "RL"

  fig.suptitle(f"{device} Parameter Exploration ({alg})", size=title_size, weight="bold")
  fig.supxlabel("Parameter Range", size=axis_size, weight="bold")
  fig.supylabel("Probability", size=axis_size, weight="bold")
  plt.show()


def pareto_front(opt, device, pdf_type):
  if opt != "Leap" and opt != "RL":
    raise ValueError("Incorrect optimization type")
  
  if device == "SOT":
    color = SOT_color
  elif device == "STT":
    color = STT_color
  else:
    raise ValueError("Incorrect device type")

  probe_df = get_full_df(opt, device, pdf_type)

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

  alg = "EA" if opt == "Leap" else "RL"

  axs.scatter(df["kl_div_score"], df["energy"], color=color, s=marker_size, label="Samples")
  axs.scatter(pareto_df["kl_div_score"], pareto_df["energy"], color=pareto_color, s=marker_size, label="Pareto Front")
  axs.tick_params(axis="both", which="major", labelsize=tick_size)
  axs.yaxis.get_offset_text().set_fontsize(tick_size)
  axs.set_title(f"{device} Pareto Front ({alg})", size=title_size, weight="bold")
  axs.set_xlabel("KL Divergence Score", size=axis_size, weight="bold")
  axs.set_ylabel("Energy", size=axis_size, weight="bold")
  axs.legend(fontsize=legend_size)
  plt.show()



if __name__ == "__main__":
  # opt = "Leap"
  opt = "RL"

  device = "SOT"
  # device = "STT"

  pdf_type = "gamma"

  # top_distributions(opt, device, pdf_type, top=5)
  # parameter_heatmap(opt, device, pdf_type, top=5)
  # parameter_exploration(opt, device, pdf_type)
  # pareto_front(opt, device, pdf_type)

  csv_name = f"{device}_{opt}.csv"
  get_best_df(opt, device, pdf_type, csv_name=csv_name)