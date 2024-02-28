import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
from scipy.special import rel_entr

sys.path.append("../")
sys.path.append("../fortran_source")
from STT_model import STT_Model
from STT_RNG_Leap_SingleProc import MTJ_RNG_Problem


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
    plt.title("Pareto Front")
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
    K_295 = param["genome"][1]
    Ms_295 = param["genome"][2]
    Rp = param["genome"][3]
    eta = param["genome"][4]
    J_stt = param["genome"][5]
    t_pulse = param["genome"][6]
    t_relax = param["genome"][6]
    TMR = 3
    d = 3e-09
    tf = 1.1e-09

    while True:
      chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, eta, J_stt, t_pulse, t_relax, samples=100000, pdf_type=pdf_type)
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
    plt.title("PDF Comparison")
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



if __name__ == "__main__":
  pdf_type = "exp"
  # pdf_type = "gamma"

  # scraper(pdf_type)
  # pareto_front(pdf_type)
  # plot_pareto_distributions(pdf_type)
  plot_top_distributions(pdf_type, top=1)