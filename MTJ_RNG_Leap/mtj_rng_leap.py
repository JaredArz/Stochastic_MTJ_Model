import sys
import time
import numpy as np
from scipy.special import rel_entr
from toolz import pipe
from distributed import Client, LocalCluster

from leap_ec import test_env_var
from leap_ec.representation import Representation
from leap_ec.ops import tournament_selection, clone, evaluate, pool
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.probe import print_individual
from leap_ec.multiobjective.probe import ParetoPlotProbe2D
from leap_ec.global_vars import context
from leap_ec.multiobjective.nsga2 import generalized_nsga_2
from leap_ec.multiobjective.problems import MultiObjectiveProblem, SCHProblem

sys.path.append("../")
sys.path.append("../fortran_source")
from mtj_model import mtj_run


# Hyperparameters
DEV_SAMPLES = 2500
alpha_range = (0.01, 0.1)
Ki_range = (0.2e-3, 1e-3)
Ms_range = (0.3e6, 2e6)
Rp_range = (500, 50000)
TMR_range = (0.3, 6)
eta_range = (0.1, 0.8)
J_she_range = (0.01e12, 1e12)
t_pulse_range = (0.5e-9, 75e-9)
t_relax_range = (0.5e-9, 75e-9)


class MTJ_RNG_Problem(MultiObjectiveProblem):
  def __init__(self):
    super().__init__(maximize=(False, False))
  
  def evaluate(self, params):
    chi2, bitstream, energy_avg, countData, bitData,  xxis, exp_pdf = mtj_run(alpha=params[0], 
                                                                              Ki=params[1], 
                                                                              Ms=params[2], 
                                                                              Rp=params[3],
                                                                              TMR=params[4], 
                                                                              d=3e-09, 
                                                                              tf=1.1e-09, 
                                                                              eta=params[5], 
                                                                              J_she=params[6], 
                                                                              t_pulse=params[7], 
                                                                              t_relax=params[7], 
                                                                              samples=DEV_SAMPLES)
    
    if chi2 == None:    # Penalize when config fails
      # kl_div = np.inf
      # energy = np.inf
      kl_div = 1_000_000
      energy = 1_000_000
    else:
      kl_div = sum(rel_entr(countData, exp_pdf))
      energy = np.mean(energy_avg)
    
    fitness = [kl_div, energy]
    print(fitness)
    return fitness


def print_generation(population):
  """ Pipeline probe for echoing current generation """
  if context['leap']['generation'] % 10 == 0:
      print(f"generation: {context['leap']['generation']}")
  return population


def train():
  N = 50
  max_generation = 100
  min_fitness = 1000
  param_bounds = [alpha_range,    # alpha bounds 
                  Ki_range,       # Ki bounds 
                  Ms_range,       # Ms bounds
                  Rp_range,       # Rp bounds
                  TMR_range,      # TMR bounds
                  eta_range,      # eta bounds
                  J_she_range,    # J_she bounds
                  t_pulse_range]  # t_pulse bounds
  
  representation = Representation(initialize=create_real_vector(bounds=param_bounds))

  pipeline = [tournament_selection, # uses domination comparison in MultiObjective.worse_than()
              clone,
              mutate_gaussian(std=0.5, expected_num_mutations=1),
              evaluate,
              # print_individual, # only if you want to see every single new offspring
              pool(size=N),
              print_generation]
  
  final_pop = generalized_nsga_2(max_generations=max_generation,
                                pop_size=N,
                                problem=MTJ_RNG_Problem(),
                                representation=representation,
                                pipeline=pipeline)
  
  print(final_pop)


if __name__ == "__main__":
  train()