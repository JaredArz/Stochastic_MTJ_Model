import os
import sys
import glob
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
from SOT_model import SOT_Model
from SOT_RNG_Leap_SingleProc import MTJ_RNG_Problem


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
  path = f"{pdf_type}_results/"

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
    plt.title(f"SOT {pdf_type.capitalize()} Pareto Front")
    plt.show()
  
  return df, pareto_df


def plot_df(df, graph_name, param_name, pdf_type):
  params = []
  for _, row in df.iterrows():
    genome = row["genome"]
    kl_div = float(row["kl_div"])
    energy = float(row["energy"])
    param = {"genome": genome, "kl_div": kl_div, "energy": energy}
    params.append(param)
  
  for i, param in enumerate(params):
    print(f"{i+1} of {len(params)}")
    alpha = param["genome"][0]
    Ki = param["genome"][1]
    Ms = param["genome"][2]
    Rp = param["genome"][3]
    eta = param["genome"][4]
    J_she = param["genome"][5]
    t_pulse = param["genome"][6]
    t_relax = param["genome"][6]
    TMR = 3
    d = 3e-09
    tf = 1.1e-09

    while True:
      chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = SOT_Model(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=100000, pdf_type=pdf_type)
      if chi2 != None:
        break
    
    kl_div_score = sum(rel_entr(countData, pdf))
    energy = np.mean(energy_avg)
    param["kl_div"] = kl_div_score
    param["energy"] = energy

    plt.plot(xxis, countData, color="red", label="Actual PDF")
    plt.plot(xxis, pdf,'k--', label="Expected PDF")
    plt.xlabel("Generated Number")
    plt.ylabel("Normalized")
    plt.title(f"SOT {pdf_type.capitalize()} PDF Comparison")
    plt.legend()
    plt.savefig(f"graphs/{graph_name}_{i}.png")
    plt.close()

    with open(f"parameters/{param_name}_{i}.pkl", "wb") as file:
      pickle.dump(param, file)


def plot_pareto_distributions(pdf_type):
  df, pareto_df = pareto_front(pdf_type, False)
  print(pareto_df)
  plot_df(pareto_df, graph_name=f"pareto_dist_{pdf_type}", param_name=f"pareto_params_{pdf_type}", pdf_type=pdf_type)

  
def plot_top_distributions(pdf_type, top=10):
  df, pareto_df = pareto_front(pdf_type, False)
  df = df.sort_values(by="kl_div").head(top)
  plot_df(df, graph_name=f"top_dist_{pdf_type}", param_name=f"top_params_{pdf_type}", pdf_type=pdf_type)


def get_norm(range):
  norm = matplotlib.colors.Normalize(vmin=range[0], vmax=range[1])
  return norm


def graph_param_values(pdf_type, top=10):
  df = scraper(pdf_type)
  df = df.sort_values(by="kl_div").head(top)

  y_labels = ["alpha", "Ki", "Ms", "Rp", "eta", "J_she", "t_pulse", "t_relax"]
  x_labels = []
  for i in range(len(df)):
    x_labels.append(f"config_{i}")

  norm_alpha = get_norm(param_ranges["alpha"])
  norm_Ki = get_norm(param_ranges["Ki"])
  norm_Ms = get_norm(param_ranges["Ms"])
  norm_Rp = get_norm(param_ranges["Rp"])
  norm_eta = get_norm(param_ranges["eta"])
  norm_J_she = get_norm(param_ranges["J_she"])
  norm_t_pulse = get_norm(param_ranges["t_pulse"])
  norm_t_relax = get_norm(param_ranges["t_relax"])
  
  fig, ax = plt.subplots()
  ax.set_xticks(np.arange(0,len(x_labels),1))
  ax.set_yticks(np.arange(0,len(y_labels),1))
  ax.set_xticklabels(x_labels)
  ax.set_yticklabels(y_labels)
  ax.set_xlabel("Config ID", size=14, weight="bold")
  ax.set_ylabel("Parameters", size=14, weight="bold")
  ax.set_title(f"SOT {pdf_type.capitalize()} Parameter Combinations", size=16, weight="bold")

  i = 0
  s = 2500
  marker = "s"
  color = "coolwarm"
  cmap = matplotlib.cm.get_cmap(color)
  for _, row in df.iterrows():
    alpha = row["genome"][0]
    Ki = row["genome"][1]
    Ms = row["genome"][2]
    Rp = row["genome"][3]
    eta = row["genome"][4]
    J_she = row["genome"][5]
    t_pulse = row["genome"][6]
    t_relax = row["genome"][6]

    ax.scatter(i, 0, s=s, marker=marker, c=cmap(norm_alpha(alpha)))
    ax.text(i, 0, f"{alpha:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 1, s=s, marker=marker, c=cmap(norm_Ki(Ki)))
    ax.text(i, 1, f"{Ki:.2E}", horizontalalignment="center", verticalalignment="center")
    
    ax.scatter(i, 2, s=s, marker=marker, c=cmap(norm_Ms(Ms)))
    ax.text(i, 2, f"{Ms:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 3, s=s, marker=marker, c=cmap(norm_Rp(Rp)))
    ax.text(i, 3, f"{Rp:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 4, s=s, marker=marker, c=cmap(norm_eta(eta)))
    ax.text(i, 4, f"{eta:.2E}", horizontalalignment="center", verticalalignment="center")
    
    ax.scatter(i, 5, s=s, marker=marker, c=cmap(norm_J_she(J_she)))
    ax.text(i, 5, f"{J_she:.2E}", horizontalalignment="center", verticalalignment="center")

    ax.scatter(i, 6, s=s, marker=marker, c=cmap(norm_t_pulse(t_pulse)))
    ax.text(i, 6, f"{t_pulse:.2E}", horizontalalignment="center", verticalalignment="center")
    
    ax.scatter(i, 7, s=s, marker=marker, c=cmap(norm_t_relax(t_relax)))
    ax.text(i, 7, f"{t_relax:.2E}", horizontalalignment="center", verticalalignment="center")

    i += 1
  
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  cbar = fig.colorbar(sm)
  cbar.ax.yaxis.set_ticks([0,1])
  cbar.ax.set_yticklabels(["Min", "Max"])
  
  plt.tight_layout()
  plt.show()



if __name__ == "__main__":
  # pdf_type = "exp"
  pdf_type = "gamma"

  # scraper(pdf_type)
  # pareto_front(pdf_type)
  # plot_pareto_distributions(pdf_type)
  # plot_top_distributions(pdf_type, top=10)
  graph_param_values(pdf_type, top=10)