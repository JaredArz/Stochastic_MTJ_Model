import os
import sys
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from toolz import pipe
from distributed import Client, LocalCluster

from leap_ec.representation import Representation
from leap_ec.ops import tournament_selection, clone, evaluate, pool
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.probe import print_individual
from leap_ec.multiobjective.problems import MultiObjectiveProblem, SCHProblem
from leap_ec.multiobjective.asynchronous import steady_state_nsga_2

sys.path.append("../")
sys.path.append("../fortran_source")
from SOT_model import SOT_Model


# Hyperparameters
POP_SIZE = 50
MAX_BIRTHS = 5000
N_WORKERS = 15

DEV_SAMPLES = 2500
alpha_range = (0.01, 0.1)
Ki_range = (0.2e-3, 1e-3)
Ms_range = (0.3e6, 2e6)
Rp_range = (500, 50000)
eta_range = (0.1, 2)
J_she_range = (0.01e12, 5e12)
t_pulse_range = (0.5e-9, 75e-9)
t_relax_range = (0.5e-9, 75e-9)


class MTJ_RNG_Problem(MultiObjectiveProblem):
  def __init__(self, pdf_type):
    super().__init__(maximize=(False, False))
    self.pdf_type = pdf_type
  
  def evaluate(self, params):
    chi2, bitstream, energy_avg, countData, bitData,  xxis, exp_pdf = SOT_Model(alpha=params[0], 
                                                                                Ki=params[1], 
                                                                                Ms=params[2], 
                                                                                Rp=params[3],
                                                                                TMR=3, 
                                                                                d=3e-09, 
                                                                                tf=1.1e-09, 
                                                                                eta=params[4], 
                                                                                J_she=params[5], 
                                                                                t_pulse=params[6], 
                                                                                t_relax=params[6], 
                                                                                samples=DEV_SAMPLES,
                                                                                pdf_type=self.pdf_type)
    
    if chi2 == None:    # Penalize when config fails
      # kl_div = np.inf
      # energy = np.inf
      kl_div = 1_000_000
      energy = 1_000_000
    else:
      kl_div = sum(rel_entr(countData, exp_pdf))
      energy = np.mean(energy_avg)
    
    fitness = [kl_div, energy]
    return fitness


def train(pdf_type, runID):
  param_bounds = [alpha_range,    # alpha bounds 
                  Ki_range,       # Ki bounds 
                  Ms_range,       # Ms bounds
                  Rp_range,       # Rp bounds
                  eta_range,      # eta bounds
                  J_she_range,    # J_she bounds
                  t_pulse_range]  # t_pulse bounds
  
  cluster = LocalCluster(n_workers=N_WORKERS)
  client = Client(cluster)

  representation = Representation(initialize=create_real_vector(bounds=param_bounds))
  
  pipeline = [tournament_selection, # uses domination comparison in MultiObjective.worse_than()
              clone,
              mutate_gaussian(std=0.5, bounds=param_bounds, expected_num_mutations=1),
              pool(size=1)]
  
  final_pop = steady_state_nsga_2(client, MAX_BIRTHS,
                   pop_size=POP_SIZE, init_pop_size=POP_SIZE,
                   problem=MTJ_RNG_Problem(pdf_type),
                   representation=representation,
                   offspring_pipeline=pipeline)
  
  result_dir = f"{pdf_type}_results"
  os.makedirs(result_dir, exist_ok=True)
  with open(f"{result_dir}/{pdf_type}_{runID}.pkl", "wb") as file:
    pickle.dump(final_pop, file)


def analyze_results(pdf_type, runID):
  with open(f"{pdf_type}_results/{pdf_type}_{runID}.pkl", "rb") as file:
    data = pickle.load(file)

  df = pd.DataFrame([(x.genome, x.fitness[0], x.fitness[1], x.rank, x.distance) for x in data])
  df.columns = ["genome","kl_div","energy","rank","distance"]
  print(df.iloc[0])
  print()
  print(df.iloc[0]["genome"])

  # Plot Pareto Front
  df.plot(x="kl_div", y="energy", kind="scatter")
  plt.show()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ID", required=True, help="run ID", type=int)
  args = parser.parse_args()
  ID = args.ID
  
  # pdf_type = "exp"
  pdf_type = "gamma"
  
  train(pdf_type, ID)
  # analyze_results(pdf_type, ID)