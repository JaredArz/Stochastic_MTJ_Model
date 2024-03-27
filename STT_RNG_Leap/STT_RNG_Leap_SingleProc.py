import os
import sys
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

from leap_ec import probe
from leap_ec.global_vars import context
from leap_ec.representation import Representation
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.ops import tournament_selection, clone, evaluate, pool
from leap_ec.multiobjective.nsga2 import generalized_nsga_2
from leap_ec.multiobjective.problems import MultiObjectiveProblem

sys.path.append("../")
sys.path.append("../fortran_source")
from STT_model import STT_Model


# Hyperparameters
# POP_SIZE = 50
# MAX_GENERATION = 50
POP_SIZE = 1
MAX_GENERATION = 1

DEV_SAMPLES = 2500
alpha_range = (0.01, 0.1)
K_295_range = (0.2e-3, 1e-3)
Ms_295_range = (0.3e6, 2e6)
Rp_range = (500, 50000)
t_pulse_range = (0.5e-9, 75e-9)
t_relax_range = (0.5e-9, 75e-9)


class MTJ_RNG_Problem(MultiObjectiveProblem):
  def __init__(self, pdf_type):
    super().__init__(maximize=(False, False))
    self.pdf_type = pdf_type
  
  def evaluate(self, params):
    chi2, bitstream, energy_avg, countData, bitData,  xxis, exp_pdf = STT_Model(alpha=params[0], 
                                                                                K_295=params[1], 
                                                                                Ms_295=params[2], 
                                                                                Rp=params[3],
                                                                                TMR=3, 
                                                                                d=3e-09, 
                                                                                tf=1.1e-09,
                                                                                t_pulse=params[4], 
                                                                                t_relax=params[4], 
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


def print_generation(population):
  fitnesses = [genome.fitness for genome in population]
  print("Generation ", context["leap"]["generation"], "Min Fitness ", min(fitnesses))
  return population


def train(pdf_type, runID):
  probe_output_dir = f"STT_{pdf_type}/probe_output"
  os.makedirs(probe_output_dir, exist_ok=True)
  stream_out = open(f"{probe_output_dir}/probe_output_{runID}.csv", "w")

  param_bounds = [alpha_range,    # alpha bounds 
                  K_295_range,    # K_295 bounds 
                  Ms_295_range,   # Ms_295 bounds
                  Rp_range,       # Rp bounds
                  t_pulse_range]  # t_pulse bounds
  
  representation = Representation(initialize=create_real_vector(bounds=param_bounds))

  pipeline = [tournament_selection, # uses domination comparison in MultiObjective.worse_than()
              clone,
              mutate_gaussian(std=0.5, bounds=param_bounds, expected_num_mutations="isotropic"),
              evaluate,
              pool(size=POP_SIZE),
              probe.AttributesCSVProbe(stream=stream_out, do_fitness=True, do_genome=True),
              print_generation]
  
  final_pop = generalized_nsga_2(max_generations=MAX_GENERATION,
                                pop_size=POP_SIZE,
                                problem=MTJ_RNG_Problem(pdf_type),
                                representation=representation,
                                pipeline=pipeline)
  
  result_dir = f"STT_{pdf_type}/results"
  os.makedirs(result_dir, exist_ok=True)
  with open(f"{result_dir}/{pdf_type}_{runID}.pkl", "wb") as file:
    pickle.dump(final_pop, file)


def analyze_results(pdf_type, runID):
  with open(f"STT_{pdf_type}/results/{pdf_type}_{runID}.pkl", "rb") as file:
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