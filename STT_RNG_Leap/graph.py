import os
import sys
import glob
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from paretoset import paretoset
from scipy.special import rel_entr
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

sys.path.append("../")
sys.path.append("../fortran_source")
from STT_model import STT_Model
from STT_RNG_Leap_SingleProc import MTJ_RNG_Problem


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


def scraper(pdf_type, csv=False):
  path = f"STT_{pdf_type}/results/"

  dataframes = []
  for i, file in enumerate(glob.glob(os.path.join(path, '*.pkl'))):
    with open(file, "rb") as f:
      data = pickle.load(f)
      df = pd.DataFrame([(x.genome, x.fitness[0], x.fitness[1], x.rank, x.distance) for x in data])
      df.columns = ["genome","kl_div","energy","rank","distance"]
      dataframes.append(df)

  df = pd.concat(dataframes, axis=0)
  if csv:
    csv_filename = f"{pdf_type}_results.csv"
    df.to_csv(csv_filename, encoding='utf-8', index=False)
  
  return df


def pareto_front(pdf_type, plot=True):
  df = scraper(pdf_type)
  pareto_df = df[["kl_div", "energy"]]

  # Get pareto front values
  mask = paretoset(pareto_df, sense=["min", "min"])
  pareto_df = df[mask]

  # Plot pareto front
  if plot == True:
    ax = df.plot.scatter(x="kl_div", y="energy", c="blue", label="Sample")
    pareto_df.plot.scatter(x="kl_div", y="energy", c="red", ax=ax, label="Pareto Front")
    plt.title(f"STT {pdf_type.capitalize()} Pareto Front")
    plt.show()
  
  return df, pareto_df


def plot_df(df, graph_name, param_name, pdf_type):
  os.makedirs(f"STT_{pdf_type}/graphs", exist_ok=True)
  os.makedirs(f"STT_{pdf_type}/parameters", exist_ok=True)
  
  params = []
  for _, row in df.iterrows():
    genome = row["genome"]
    kl_div = float(row["kl_div"])
    energy = float(row["energy"])
    param = {"genome": genome, "kl_div": kl_div, "energy": energy, "xxis": None, "countData": None}
    params.append(param)
  
  for i, param in enumerate(params):
    print(f"{i+1} of {len(params)}")
    alpha = param["genome"][0]
    K_295 = param["genome"][1]
    Ms_295 = param["genome"][2]
    Rp = param["genome"][3]
    t_pulse = param["genome"][4]
    t_relax = param["genome"][4]
    TMR = 3
    d = 3e-09
    tf = 1.1e-09

    while True:
      chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, t_pulse, t_relax, samples=100000, pdf_type=pdf_type)
      if chi2 != None:
        break
    
    kl_div_score = sum(rel_entr(countData, pdf))
    energy = np.mean(energy_avg)
    param["kl_div"] = kl_div_score
    param["energy"] = energy
    param["xxis"] = xxis
    param["countData"] = countData

    plt.plot(xxis, countData, color="blueviolet", label="STT PDF")
    plt.plot(xxis, pdf, color="dimgray", linestyle="dashed", label="Expected PDF")
    plt.xlabel("Generated Number")
    plt.ylabel("Normalized")
    plt.title(f"STT {pdf_type.capitalize()} PDF Comparison")
    plt.legend()
    plt.savefig(f"STT_{pdf_type}/graphs/{graph_name}_{i}.png")
    plt.close()

    with open(f"STT_{pdf_type}/parameters/{param_name}_{i}.pkl", "wb") as file:
      pickle.dump(param, file)


def plot_pareto_distributions(pdf_type):
  df, pareto_df = pareto_front(pdf_type, False)
  plot_df(pareto_df, graph_name=f"{pdf_type}_pareto", param_name=f"{pdf_type}_pareto", pdf_type=pdf_type)

  
def plot_top_distributions(pdf_type, top=10):
  df, pareto_df = pareto_front(pdf_type, False)
  df = df.sort_values(by="kl_div").head(top)
  plot_df(df, graph_name=f"{pdf_type}_top", param_name=f"{pdf_type}_top", pdf_type=pdf_type)


def get_norm(range):
  norm = matplotlib.colors.Normalize(vmin=range[0], vmax=range[1])
  return norm


def graph_param_values(pdf_type, top=10):
  df = scraper(pdf_type)
  df = df.sort_values(by="kl_div").head(top)

  y_labels = ["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax"]
  x_labels = []
  for i in range(len(df)):
    x_labels.append(f"config_{i}")

  norm_alpha = get_norm(param_ranges["alpha"])
  norm_K_295 = get_norm(param_ranges["K_295"])
  norm_Ms_295 = get_norm(param_ranges["Ms_295"])
  norm_Rp = get_norm(param_ranges["Rp"])
  norm_t_pulse = get_norm(param_ranges["t_pulse"])
  norm_t_relax = get_norm(param_ranges["t_relax"])
  
  fig, ax = plt.subplots()
  ax.set_xticks(np.arange(0,len(x_labels),1))
  ax.set_yticks(np.arange(0,len(y_labels),1))
  ax.set_xticklabels(x_labels)
  ax.set_yticklabels(y_labels)
  ax.set_xlabel("Config ID", size=14, weight="bold")
  ax.set_ylabel("Parameters", size=14, weight="bold")
  ax.set_title(f"STT {pdf_type.capitalize()} Parameter Combinations (LEAP)", size=16, weight="bold")

  i = 0
  s = 2500
  marker = "s"
  color = "coolwarm"
  cmap = matplotlib.cm.get_cmap(color)
  for _, row in df.iterrows():
    alpha = row["genome"][0]
    K_295 = row["genome"][1]
    Ms_295 = row["genome"][2]
    Rp = row["genome"][3]
    t_pulse = row["genome"][4]
    t_relax = row["genome"][4]

    ax.scatter(i, 0, s=s, marker=marker, c=cmap(norm_alpha(alpha)))
    ax.text(i, 0, f"{alpha:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 1, s=s, marker=marker, c=cmap(norm_K_295(K_295)))
    ax.text(i, 1, f"{K_295:.2E}", horizontalalignment="center", verticalalignment="center")
    
    ax.scatter(i, 2, s=s, marker=marker, c=cmap(norm_Ms_295(Ms_295)))
    ax.text(i, 2, f"{Ms_295:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 3, s=s, marker=marker, c=cmap(norm_Rp(Rp)))
    ax.text(i, 3, f"{Rp:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 4, s=s, marker=marker, c=cmap(norm_t_pulse(t_pulse)))
    ax.text(i, 4, f"{t_pulse:.2E}", horizontalalignment="center", verticalalignment="center")
    
    ax.scatter(i, 5, s=s, marker=marker, c=cmap(norm_t_relax(t_relax)))
    ax.text(i, 5, f"{t_relax:.2E}", horizontalalignment="center", verticalalignment="center")

    i += 1
  
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  cbar = fig.colorbar(sm)
  cbar.ax.yaxis.set_ticks([0,1])
  cbar.ax.set_yticklabels(["Min", "Max"])
  
  plt.tight_layout()
  plt.show()


def graph_param_exploration(pdf_type):
  path = f"STT_{pdf_type}/probe_output/"

  main_df = pd.DataFrame(columns=["alpha", "K_295", "Ms_295", "Rp", "t_pulse", "t_relax"])
  for i, csvFile in enumerate(glob.glob(os.path.join(path, "*.csv"))):
    df = pd.read_csv(csvFile)

    for genome in df["genome"]:
      genome = ast.literal_eval(genome)
      genome.append(genome[-1])
      main_df.loc[len(main_df)] = genome

  num_bins = 15
  samples = len(main_df)

  alpha_counts, _ = np.histogram(main_df["alpha"].to_list(), bins=num_bins)
  alpha_counts = alpha_counts/samples
  alpha_xxis = np.linspace(param_ranges["alpha"][0], param_ranges["alpha"][1], num_bins)

  K_295_counts, _ = np.histogram(main_df["K_295"].to_list(), bins=num_bins)
  K_295_counts = K_295_counts/samples
  K_295_xxis = np.linspace(param_ranges["K_295"][0], param_ranges["K_295"][1], num_bins)

  Ms_295_counts, _ = np.histogram(main_df["Ms_295"].to_list(), bins=num_bins)
  Ms_295_counts = Ms_295_counts/samples
  Ms_295_xxis = np.linspace(param_ranges["Ms_295"][0], param_ranges["Ms_295"][1], num_bins)

  Rp_counts, _ = np.histogram(main_df["Rp"].to_list(), bins=num_bins)
  Rp_counts = Rp_counts/samples
  Rp_xxis = np.linspace(param_ranges["Rp"][0], param_ranges["Rp"][1], num_bins)

  t_relax_counts, _ = np.histogram(main_df["t_relax"].to_list(), bins=num_bins)
  t_relax_counts = t_relax_counts/samples
  t_relax_xxis = np.linspace(param_ranges["t_relax"][0], param_ranges["t_relax"][1], num_bins)

  t_pulse_counts, _ = np.histogram(main_df["t_pulse"].to_list(), bins=num_bins)
  t_pulse_counts = t_pulse_counts/samples
  t_pulse_xxis = np.linspace(param_ranges["t_pulse"][0], param_ranges["t_pulse"][1], num_bins)

  fig, axs = plt.subplots(2, 3, sharey=True, layout="tight")

  axs[0,0].plot(alpha_xxis, alpha_counts, color="blueviolet")
  axs[0,0].set_xticks([param_ranges["alpha"][0], param_ranges["alpha"][1]], visible=True, rotation="horizontal")
  axs[0,0].set_title("alpha")

  axs[0,1].plot(K_295_xxis, K_295_counts, color="blueviolet")
  axs[0,1].set_xticks([param_ranges["K_295"][0], param_ranges["K_295"][1]], visible=True, rotation="horizontal")
  axs[0,1].set_title("K_295")

  axs[0,2].plot(Ms_295_xxis, Ms_295_counts, color="blueviolet")
  axs[0,2].set_xticks([param_ranges["Ms_295"][0], param_ranges["Ms_295"][1]], visible=True, rotation="horizontal")
  axs[0,2].set_title("Ms_295")

  axs[1,0].plot(Rp_xxis, Rp_counts, color="blueviolet")
  axs[1,0].set_xticks([param_ranges["Rp"][0], param_ranges["Rp"][1]], visible=True, rotation="horizontal")
  axs[1,0].set_title("Rp")

  axs[1,1].plot(t_pulse_xxis, t_pulse_counts, color="blueviolet")
  axs[1,1].set_xticks([param_ranges["t_pulse"][0], param_ranges["t_pulse"][1]], visible=True, rotation="horizontal")
  axs[1,1].set_title("t_pulse")

  axs[1,2].plot(t_relax_xxis, t_relax_counts, color="blueviolet")
  axs[1,2].set_xticks([param_ranges["t_relax"][0], param_ranges["t_relax"][1]], visible=True, rotation="horizontal")
  axs[1,2].set_title("t_relax")
  
  fig.supxlabel("Parameter Range", size=18, weight="bold")
  fig.supylabel("Probability", size=18, weight="bold", x=-0.001)
  fig.suptitle(f"STT {pdf_type.capitalize()} Parameter Exploration (RL)", size=20, weight="bold")
  plt.show()



if __name__ == "__main__":
  # pdf_type = "exp"
  pdf_type = "gamma"

  # scraper(pdf_type)
  # pareto_front(pdf_type)
  # plot_pareto_distributions(pdf_type)
  plot_top_distributions(pdf_type, top=10)
  # graph_param_values(pdf_type, top=10)
  # graph_param_exploration(pdf_type)