import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

sys.path.append("../")
sys.path.append("../fortran_source")
from mtj_model import mtj_run
from SOT_RNG_Leap_SingleProc import MTJ_RNG_Problem


def scraper():
  dataframes = []
  path = "leap_results/"
  for i, file in enumerate(glob.glob(os.path.join(path, '*.pkl'))):
    with open(file, "rb") as f:
      data = pickle.load(f)
      df = pd.DataFrame([(x.genome, x.fitness[0], x.fitness[1], x.rank, x.distance) for x in data])
      df.columns = ["genome","kl_div","energy","rank","distance"]
      dataframes.append(df)

  df = pd.concat(dataframes, axis=0)
  df.to_csv("leap_results.csv", encoding='utf-8', index=False)
  return df


def pareto_front(plot=True):
  df = scraper()
  pareto_df = df[["kl_div", "energy"]]

  # Get pareto front values
  mask = paretoset(pareto_df, sense=["min", "min"])
  pareto_df = pareto_df[mask]

  # Plot pareto front
  if plot == True:
    ax = df.plot.scatter(x="kl_div", y="energy", c="blue")
    pareto_df.plot.scatter(x="kl_div", y="energy", c="red", ax=ax)
    plt.show()
  
  return df, pareto_df


def plot_distributions():
  df, pareto_df = pareto_front(False)

  params = []
  for _, row in pareto_df.iterrows():
    temp_df = df[(df["kl_div"] == row["kl_div"]) & (df["energy"] == row["energy"])]
    genome = temp_df["genome"].to_list()[0]
    kl_div = float(temp_df["kl_div"])
    energy = float(temp_df["energy"])
    param = {"genome": genome, "kl_div": kl_div, "energy": energy}
    params.append(param)
  
  for i, param in enumerate(params):
    print(f"{i} of {len(params)}")
    alpha = param["genome"][0]
    Ki = param["genome"][1]
    Ms = param["genome"][2]
    Rp = param["genome"][3]
    TMR = param["genome"][4]
    eta = param["genome"][5]
    J_she = param["genome"][6]
    t_pulse = param["genome"][7]
    t_relax = param["genome"][7]
    d = 3e-09
    tf = 1.1e-09

    while True:
      chi2, _, _, countData, _, xxis, exp_pdf = mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=100000)
      if chi2 != None:
        break

    plt.plot(xxis, countData, color="red", label="Actual PDF")
    plt.plot(xxis, exp_pdf,'k--', label="Expected PDF")
    plt.xlabel("Generated Number")
    plt.ylabel("Normalized")
    plt.title("PDF Comparison")
    plt.legend()
    plt.savefig(f"graphs/distribution_{i}.png")
    plt.close()

    with open(f"parameters/params_{i}.pkl", "wb") as file:
      pickle.dump(param, file)



if __name__ == "__main__":
  # scraper()
  # pareto_front()
  plot_distributions()
