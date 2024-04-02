import os
import sys
import glob
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from paretoset import paretoset
from scipy import stats
from scipy.special import rel_entr
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

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


def get_df(pdf_type):
  SOT_df = pd.DataFrame(columns=["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax", "kl_div_score", "energy", "xxis", "countData"])
  STT_df = pd.DataFrame(columns=["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax", "kl_div_score", "energy", "xxis", "countData"])

  SOT_param_path = f"../SOT_RNG_RL/SOT_{pdf_type}/parameters"
  STT_param_path = f"../STT_RNG_RL/STT_{pdf_type}/parameters"

  for i, pklFile in enumerate(glob.glob(os.path.join(SOT_param_path, "*.pkl"))):
    with open(pklFile, "rb") as f:
      data = pickle.load(f)
      
      alpha = data["alpha"]
      Ki = data["Ki"]
      Ms = data["Ms"]
      Rp = data["Rp"]
      eta = data["eta"]
      J_she = data["J_she"]
      t_pulse = data["t_pulse"]
      t_relax = data["t_relax"]
      kl_div_score = data["kl_div_score"]
      energy = data["energy"]
      xxis = data["xxis"]
      countData = data["countData"]

      row = [alpha, Ki, Ms, Rp, eta, J_she, t_pulse, t_relax, kl_div_score, energy, xxis, countData]
      SOT_df.loc[len(SOT_df)] = row
  
  for i, pklFile in enumerate(glob.glob(os.path.join(STT_param_path, "*.pkl"))):
    with open(pklFile, "rb") as f:
      data = pickle.load(f)
      
      alpha = data["alpha"]
      K_295 = data["K_295"]
      Ms_295 = data["Ms_295"]
      Rp = data["Rp"]
      t_pulse = data["t_pulse"]
      t_relax = data["t_relax"]
      kl_div_score = data["kl_div_score"]
      energy = data["energy"]
      xxis = data["xxis"]
      countData = data["countData"]

      row = [alpha, K_295, Ms_295, Rp, t_pulse, t_relax, kl_div_score, energy, xxis, countData]
      STT_df.loc[len(STT_df)] = row

  SOT_df = SOT_df.sort_values(by="kl_div_score")
  SOT_df.reset_index(drop=True, inplace=True)

  STT_df = STT_df.sort_values(by="kl_div_score")
  STT_df.reset_index(drop=True, inplace=True)

  return SOT_df, STT_df



def top_distributions(pdf_type, top=5):
  xxis, target_pdf, prng_pdf = prng_dist(samples=100_000)
  SOT_df, STT_df = get_df(pdf_type)
  
  SOT_df_top = SOT_df.head(top)
  STT_df_top = STT_df.head(top)

  sot_pdf_arr = []
  for i, row in SOT_df_top.iterrows():
    sot_pdf_arr.append(row["countData"])
  
  stt_pdf_arr = []
  for i, row in STT_df_top.iterrows():
    stt_pdf_arr.append(row["countData"])

  sot_pdf_mean = np.average(sot_pdf_arr, axis = 0)
  sot_pdf_std = np.std(sot_pdf_arr, axis = 0)

  stt_pdf_mean = np.average(stt_pdf_arr, axis = 0)
  stt_pdf_std = np.std(stt_pdf_arr, axis = 0)

  fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, layout="constrained")

  axs[0].plot(xxis, sot_pdf_mean, color=SOT_color2, linewidth=1, label="SOT Top 5 Avg.")
  axs[0].fill_between(xxis, sot_pdf_mean-sot_pdf_std, sot_pdf_mean+sot_pdf_std, alpha=0.5, facecolor=SOT_color2, edgecolor=SOT_color2)
  axs[0].plot(xxis, sot_pdf_arr[0], color=SOT_color, linewidth=1.5, label="SOT Best")
  axs[0].plot(xxis, prng_pdf, color=prng_pdf_color, linewidth=1.5, label="PRNG")
  axs[0].plot(xxis, target_pdf, color=target_pdf_color, linewidth=1.5, linestyle="dashed", label="Target")
  axs[0].set_title("SOT PDF Comparison", weight="bold")
  axs[0].legend()

  axs[1].plot(xxis, stt_pdf_mean, color=STT_color2, linewidth=1, label="STT Top 5 Avg.")
  axs[1].fill_between(xxis, stt_pdf_mean-stt_pdf_std, stt_pdf_mean+stt_pdf_std, alpha=0.5, facecolor=STT_color2, edgecolor=STT_color2)
  axs[1].plot(xxis, stt_pdf_arr[0], color=STT_color, linewidth=1.5, label="STT Best")
  axs[1].plot(xxis, prng_pdf, color=prng_pdf_color, linewidth=1.5, label="PRNG")
  axs[1].plot(xxis, target_pdf, color=target_pdf_color, linewidth=1.5, linestyle="dashed", label="Target")
  axs[1].set_title("STT PDF Comparison", weight="bold")
  axs[1].legend()

  fig.supxlabel("Generated Number", size=14, weight="bold")
  fig.supylabel("Probability", size=14, weight="bold")
  fig.suptitle(f"Top Distribution Results (RL)", size=16, weight="bold")
  plt.show()


def get_norm(range):
  norm = matplotlib.colors.Normalize(vmin=range[0], vmax=range[1])
  return norm


def parameter_heatmap(pdf_type, top=5):
  SOT_df, STT_df = get_df(pdf_type)

  SOT_df_top = SOT_df.head(top)
  STT_df_top = STT_df.head(top)

  SOT_ylabels = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax"]
  SOT_xlabels = []
  for i in range(len(SOT_df_top)):
    SOT_xlabels.append(f"config_{i}")

  STT_ylabels = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax"]
  STT_xlabels = []
  for i in range(len(STT_df_top)):
    STT_xlabels.append(f"config_{i}")
  
  norm_alpha = get_norm(param_ranges["alpha"])
  norm_Ki = get_norm(param_ranges["Ki"])
  norm_Ms = get_norm(param_ranges["Ms"])
  norm_K_295 = get_norm(param_ranges["K_295"])
  norm_Ms_295 = get_norm(param_ranges["Ms_295"])
  norm_Rp = get_norm(param_ranges["Rp"])
  norm_eta = get_norm(param_ranges["eta"])
  norm_J_she = get_norm(param_ranges["J_she"])
  norm_t_pulse = get_norm(param_ranges["t_pulse"])
  norm_t_relax = get_norm(param_ranges["t_relax"])

  s = 2500
  size = 8
  marker = "s"
  color = "coolwarm"
  cmap = matplotlib.cm.get_cmap(color)
  fig, axs = plt.subplots(1, 2, layout="constrained")

  axs[0].set_xticks(np.arange(0,len(SOT_xlabels),1))
  axs[0].set_yticks(np.arange(0,len(SOT_ylabels),1))
  axs[0].set_xticklabels(SOT_xlabels)
  axs[0].set_yticklabels(SOT_ylabels)
  axs[0].set_title("SOT Top Parameter Configurations", weight="bold")

  for i, row in SOT_df_top.iterrows():
    alpha = row["alpha"]
    Ki = row["Ki"]
    Ms = row["Ms"]
    Rp = row["Rp"]
    eta = row["eta"]
    J_she = row["J_she"]
    t_pulse = row["t_pulse"]
    t_relax = row["t_relax"]

    axs[0].scatter(i, 0, s=s, marker=marker, c=cmap(norm_alpha(alpha)))
    axs[0].text(i, 0, f"{alpha:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_alpha(alpha) > 0.25 and norm_alpha(alpha) < 0.75) else "white")
    axs[0].scatter(i, 1, s=s, marker=marker, c=cmap(norm_Ki(Ki)))
    axs[0].text(i, 1, f"{Ki:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_Ki(Ki) > 0.25 and norm_Ki(Ki) < 0.75) else "white")
    axs[0].scatter(i, 2, s=s, marker=marker, c=cmap(norm_Ms(Ms)))
    axs[0].text(i, 2, f"{Ms:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_Ms(Ms) > 0.25 and norm_Ms(Ms) < 0.75) else "white")
    axs[0].scatter(i, 3, s=s, marker=marker, c=cmap(norm_Rp(Rp)))
    axs[0].text(i, 3, f"{Rp:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_Rp(Rp) > 0.25 and norm_Rp(Rp) < 0.75) else "white")
    axs[0].scatter(i, 4, s=s, marker=marker, c=cmap(norm_eta(eta)))
    axs[0].text(i, 4, f"{eta:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_eta(eta) > 0.25 and norm_eta(eta) < 0.75) else "white")
    axs[0].scatter(i, 5, s=s, marker=marker, c=cmap(norm_J_she(J_she)))
    axs[0].text(i, 5, f"{J_she:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_J_she(J_she) > 0.25 and norm_J_she(J_she) < 0.75) else "white")
    axs[0].scatter(i, 6, s=s, marker=marker, c=cmap(norm_t_pulse(t_pulse)))
    axs[0].text(i, 6, f"{t_pulse:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_t_pulse(t_pulse) > 0.25 and norm_t_pulse(t_pulse) < 0.75) else "white")
    axs[0].scatter(i, 7, s=s, marker=marker, c=cmap(norm_t_relax(t_relax)))
    axs[0].text(i, 7, f"{t_relax:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_t_relax(t_relax) > 0.25 and norm_t_relax(t_relax) < 0.75) else "white")

  axs[1].set_xticks(np.arange(0,len(STT_xlabels),1))
  axs[1].set_yticks(np.arange(0,len(STT_ylabels),1))
  axs[1].set_xticklabels(STT_xlabels)
  axs[1].set_yticklabels(STT_ylabels)
  axs[1].set_title("STT Top Parameter Configurations", weight="bold")

  for i, row in STT_df_top.iterrows():
    alpha = row["alpha"]
    K_295 = row["K_295"]
    Ms_295 = row["Ms_295"]
    Rp = row["Rp"]
    t_pulse = row["t_pulse"]
    t_relax = row["t_relax"]

    axs[1].scatter(i, 0, s=s, marker=marker, c=cmap(norm_alpha(alpha)))
    axs[1].text(i, 0, f"{alpha:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_alpha(alpha) > 0.25 and norm_alpha(alpha) < 0.75) else "white")
    axs[1].scatter(i, 1, s=s, marker=marker, c=cmap(norm_K_295(K_295)))
    axs[1].text(i, 1, f"{K_295:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_K_295(K_295) > 0.25 and norm_K_295(K_295) < 0.75) else "white")
    axs[1].scatter(i, 2, s=s, marker=marker, c=cmap(norm_Ms_295(Ms_295)))
    axs[1].text(i, 2, f"{Ms_295:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_Ms_295(Ms_295) > 0.25 and norm_Ms_295(Ms_295) < 0.75) else "white")
    axs[1].scatter(i, 3, s=s, marker=marker, c=cmap(norm_Rp(Rp)))
    axs[1].text(i, 3, f"{Rp:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_Rp(Rp) > 0.25 and norm_Rp(Rp) < 0.75) else "white")
    axs[1].scatter(i, 4, s=s, marker=marker, c=cmap(norm_t_pulse(t_pulse)))
    axs[1].text(i, 4, f"{t_pulse:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_t_pulse(t_pulse) > 0.25 and norm_t_pulse(t_pulse) < 0.75) else "white")
    axs[1].scatter(i, 5, s=s, marker=marker, c=cmap(norm_t_relax(t_relax)))
    axs[1].text(i, 5, f"{t_relax:.1e}", size=size, horizontalalignment="center", verticalalignment="center",
                color = "black" if (norm_t_relax(t_relax) > 0.25 and norm_t_relax(t_relax) < 0.75) else "white")

  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  cbar = fig.colorbar(sm)
  cbar.ax.yaxis.set_ticks([0,1])
  cbar.ax.set_yticklabels(["Min", "Max"])
  fig.supxlabel("Config ID", size=14, weight="bold")
  fig.supylabel("Parameters", size=14, weight="bold")
  fig.suptitle(f"Top Parameter Combinations (RL)", size=16, weight="bold")
  plt.show()


def get_csv_df(pdf_type):
  SOT_csvFile = "../SOT_RNG_RL/SOT_Gamma_Model-timestep-6000_Results.csv"
  STT_csvFile = "../STT_RNG_RL/STT_Gamma_Model-timestep-6000_Results.csv"

  SOT_df = pd.read_csv(SOT_csvFile)
  SOT_df = SOT_df.sort_values(by="kl_div_score")
  subset = ["alpha", "Ki", "Ms", "Rp", "TMR", "eta", "J_she", "t_pulse", "t_relax", "d", "tf"]
  SOT_df.drop_duplicates(subset=subset, keep="first")
  SOT_df.reset_index(drop=True, inplace=True)

  STT_df = pd.read_csv(STT_csvFile)
  STT_df = STT_df.sort_values(by="kl_div_score")
  subset = ["alpha", "K_295", "Ms_295", "Rp", "TMR", "t_pulse", "t_relax", "d", "tf"]
  STT_df.drop_duplicates(subset=subset, keep="first")
  STT_df.reset_index(drop=True, inplace=True)

  return SOT_df, STT_df


def get_dist(df, param, bins, samples):
  counts, _ = np.histogram(df[param].to_list(), bins=bins)
  counts = counts/samples
  xxis = np.linspace(param_ranges[param][0], param_ranges[param][1], bins)

  return xxis, counts


def parameter_exploration(pdf_type):
  SOT_df, STT_df = get_csv_df(pdf_type)

  bins = 15
  SOT_samples = len(SOT_df)
  STT_samples = len(STT_df)
  
  SOT_alpha_xxis, SOT_alpha_counts = get_dist(SOT_df, "alpha", bins, SOT_samples)
  SOT_Ki_xxis, SOT_Ki_counts = get_dist(SOT_df, "Ki", bins, SOT_samples)
  SOT_Ms_xxis, SOT_Ms_counts = get_dist(SOT_df, "Ms", bins, SOT_samples)
  SOT_Rp_xxis, SOT_Rp_counts = get_dist(SOT_df, "Rp", bins, SOT_samples)
  SOT_eta_xxis, SOT_eta_counts = get_dist(SOT_df, "eta", bins, SOT_samples)
  SOT_J_she_xxis, SOT_J_she_counts = get_dist(SOT_df, "J_she", bins, SOT_samples)
  SOT_t_relax_xxis, SOT_t_relax_counts = get_dist(SOT_df, "t_relax", bins, SOT_samples)
  SOT_t_pulse_xxis, SOT_t_pulse_counts = get_dist(SOT_df, "t_pulse", bins, SOT_samples)

  STT_alpha_xxis, STT_alpha_counts = get_dist(STT_df, "alpha", bins, STT_samples)
  STT_K_295_xxis, STT_K_295_counts = get_dist(STT_df, "K_295", bins, STT_samples)
  STT_Ms_295_xxis, STT_Ms_295_counts = get_dist(STT_df, "Ms_295", bins, STT_samples)
  STT_Rp_xxis, STT_Rp_counts = get_dist(STT_df, "Rp", bins, STT_samples)
  STT_t_relax_xxis, STT_t_relax_counts = get_dist(STT_df, "t_relax", bins, STT_samples)
  STT_t_pulse_xxis, STT_t_pulse_counts = get_dist(STT_df, "t_pulse", bins, STT_samples)


  fig = plt.figure(constrained_layout=True)
  (SOT_fig, STT_fig) = fig.subfigures(1, 2)

  SOT_axs = SOT_fig.subplots(2, 4, sharey=True)
  STT_axs = STT_fig.subplots(2, 3, sharey=True)

  SOT_axs[0,0].plot(SOT_alpha_xxis, SOT_alpha_counts, color=SOT_color)
  SOT_axs[0,0].set_xticks([param_ranges["alpha"][0], param_ranges["alpha"][1]], visible=True, rotation="horizontal")
  SOT_axs[0,0].set_title("alpha")

  SOT_axs[0,1].plot(SOT_Ki_xxis, SOT_Ki_counts, color=SOT_color)
  SOT_axs[0,1].set_xticks([param_ranges["Ki"][0], param_ranges["Ki"][1]], visible=True, rotation="horizontal")
  SOT_axs[0,1].set_title("Ki")

  SOT_axs[0,2].plot(SOT_Ms_xxis, SOT_Ms_counts, color=SOT_color)
  SOT_axs[0,2].set_xticks([param_ranges["Ms"][0], param_ranges["Ms"][1]], visible=True, rotation="horizontal")
  SOT_axs[0,2].set_title("Ms")

  SOT_axs[0,3].plot(SOT_Rp_xxis, SOT_Rp_counts, color=SOT_color)
  SOT_axs[0,3].set_xticks([param_ranges["Rp"][0], param_ranges["Rp"][1]], visible=True, rotation="horizontal")
  SOT_axs[0,3].set_title("Rp")

  SOT_axs[1,0].plot(SOT_eta_xxis, SOT_eta_counts, color=SOT_color)
  SOT_axs[1,0].set_xticks([param_ranges["eta"][0], param_ranges["eta"][1]], visible=True, rotation="horizontal")
  SOT_axs[1,0].set_title("eta")

  SOT_axs[1,1].plot(SOT_J_she_xxis, SOT_J_she_counts, color=SOT_color)
  SOT_axs[1,1].set_xticks([param_ranges["J_she"][0], param_ranges["J_she"][1]], visible=True, rotation="horizontal")
  SOT_axs[1,1].set_title("J_she")

  SOT_axs[1,2].plot(SOT_t_pulse_xxis, SOT_t_pulse_counts, color=SOT_color)
  SOT_axs[1,2].set_xticks([param_ranges["t_pulse"][0], param_ranges["t_pulse"][1]], visible=True, rotation="horizontal")
  SOT_axs[1,2].set_title("t_pulse")

  SOT_axs[1,3].plot(SOT_t_relax_xxis, SOT_t_relax_counts, color=SOT_color)
  SOT_axs[1,3].set_xticks([param_ranges["t_relax"][0], param_ranges["t_relax"][1]], visible=True, rotation="horizontal")
  SOT_axs[1,3].set_title("t_relax")

  STT_axs[0,0].plot(STT_alpha_xxis, STT_alpha_counts, color=STT_color)
  STT_axs[0,0].set_xticks([param_ranges["alpha"][0], param_ranges["alpha"][1]], visible=True, rotation="horizontal")
  STT_axs[0,0].set_title("alpha")

  STT_axs[0,1].plot(STT_K_295_xxis, STT_K_295_counts, color=STT_color)
  STT_axs[0,1].set_xticks([param_ranges["K_295"][0], param_ranges["K_295"][1]], visible=True, rotation="horizontal")
  STT_axs[0,1].set_title("K_295")

  STT_axs[0,2].plot(STT_Ms_295_xxis, STT_Ms_295_counts, color=STT_color)
  STT_axs[0,2].set_xticks([param_ranges["Ms_295"][0], param_ranges["Ms_295"][1]], visible=True, rotation="horizontal")
  STT_axs[0,2].set_title("Ms_295")

  STT_axs[1,0].plot(STT_Rp_xxis, STT_Rp_counts, color=STT_color)
  STT_axs[1,0].set_xticks([param_ranges["Rp"][0], param_ranges["Rp"][1]], visible=True, rotation="horizontal")
  STT_axs[1,0].set_title("Rp")

  STT_axs[1,1].plot(STT_t_pulse_xxis, STT_t_pulse_counts, color=STT_color)
  STT_axs[1,1].set_xticks([param_ranges["t_pulse"][0], param_ranges["t_pulse"][1]], visible=True, rotation="horizontal")
  STT_axs[1,1].set_title("t_pulse")

  STT_axs[1,2].plot(STT_t_relax_xxis, STT_t_relax_counts, color=STT_color)
  STT_axs[1,2].set_xticks([param_ranges["t_relax"][0], param_ranges["t_relax"][1]], visible=True, rotation="horizontal")
  STT_axs[1,2].set_title("t_relax")
  
  # SOT_fig.set_linewidth(1)
  # SOT_fig.set_edgecolor('black')
  SOT_fig.set_facecolor("#eaebed")
  SOT_fig.suptitle("SOT", weight="bold")
  
  # STT_fig.set_linewidth(1)
  # STT_fig.set_edgecolor('black')
  STT_fig.set_facecolor("#eaebed")
  STT_fig.suptitle("STT", weight="bold")
  
  fig.supxlabel("Parameter Range", size=14, weight="bold")
  fig.supylabel("Probability", size=14, weight="bold")
  fig.suptitle(f"Parameter Exploration (RL)", size=16, weight="bold")
  plt.show()


def pareto_front(pdf_type):
  SOT_df, STT_df = get_csv_df(pdf_type)

  SOT_df = SOT_df[ (SOT_df["kl_div_score"] != 1000000) & (SOT_df["energy"] != 1000000) ]
  SOT_pareto_df = SOT_df[["kl_div_score", "energy"]]
  mask = paretoset(SOT_pareto_df, sense=["min", "min"])
  SOT_pareto_df = SOT_df[mask]

  STT_df = STT_df[ (STT_df["kl_div_score"] != 1000000) & (STT_df["energy"] != 1000000) ]
  STT_pareto_df = STT_df[["kl_div_score", "energy"]]
  mask = paretoset(STT_pareto_df, sense=["min", "min"])
  STT_pareto_df = STT_df[mask]

  fig, axs = plt.subplots(1, 2, layout="constrained")

  axs[0].scatter(SOT_df["kl_div_score"], SOT_df["energy"], color=SOT_color, label="SOT Samples")
  axs[0].scatter(SOT_pareto_df["kl_div_score"], SOT_pareto_df["energy"], color=pareto_color, label="SOT Pareto Front")
  axs[0].set_title("SOT", weight="bold")
  axs[0].legend()

  axs[1].scatter(STT_df["kl_div_score"], STT_df["energy"], color=STT_color, label="STT Samples")
  axs[1].scatter(STT_pareto_df["kl_div_score"], STT_pareto_df["energy"], color=pareto_color, label="STT Pareto Front")
  axs[1].set_title("STT", weight="bold")
  axs[1].legend()

  fig.supxlabel("KL Divergence Score", size=14, weight="bold")
  fig.supylabel("Energy", size=14, weight="bold")
  fig.suptitle(f"Pareto Front (LEAP)", size=16, weight="bold")
  plt.show()



if __name__ == "__main__":
  pdf_type = "gamma"

  # top_distributions(pdf_type, top=5)
  # parameter_heatmap(pdf_type, top=5)
  # parameter_exploration(pdf_type)
  pareto_front(pdf_type)