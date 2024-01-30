import sys
import time
import numpy as np
from scipy.special import rel_entr
from toolz import pipe
from distributed import Client, LocalCluster

from leap_ec.distrib.individual import DistributedIndividual
from leap_ec.distrib import synchronous
from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops, util
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.binary_rep.problems import ScalarProblem
from leap_ec.decoder import IdentityDecoder

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


class MTJ_RNG_Problem(ScalarProblem):
  def __init__(self):
    super().__init__(maximize=False)
  
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
      # fitness = np.inf
      fitness = 1_000_000
    else:
      w1 = 0.5
      w2 = 0.5
      kl_div = sum(rel_entr(countData, exp_pdf))
      energy = np.mean(energy_avg)
      fitness = w1*kl_div + w2*energy
    print(fitness)
    return fitness

def train():
  pop_size = 100
  max_generation = 100
  min_fitness = 1000

  client = Client()
  param_bounds = [alpha_range,    # alpha bounds 
                  Ki_range,       # Ki bounds 
                  Ms_range,       # Ms bounds
                  Rp_range,       # Rp bounds
                  TMR_range,      # TMR bounds
                  eta_range,      # eta bounds
                  J_she_range,    # J_she bounds
                  t_pulse_range]  # t_pulse bounds

  parents = DistributedIndividual.create_population(pop_size,
                                                    initialize=create_real_vector(bounds=param_bounds),
                                                    decoder=IdentityDecoder(),
                                                    problem=MTJ_RNG_Problem())

  start_time = time.time()

  parents = synchronous.eval_population(parents,client=client)
  for g in range(max_generation):
    offspring = pipe(parents,
                    ops.tournament_selection,
                    ops.clone,
                    # mutate_gaussian(std=0.001, hard_bounds=(0, 1), expected_num_mutations=1),
                    mutate_gaussian(std=0.5, expected_num_mutations=1),
                    ops.UniformCrossover(),
                    synchronous.eval_pool(client=client, size=len(parents)),
                    ops.elitist_survival(parents=parents))  # accumulate offspring
          
    parents = offspring    
    fitnesses = [genome.fitness for genome in offspring]
    print("Generation ", g, "Min Fitness ", min(fitnesses))

    for genome in offspring:
      if (genome.fitness < min_fitness):
        min_fitness = genome.fitness
        best_individual = genome
              
  end_time = time.time()
  print("Time elapsed:", end_time-start_time)
  print(best_individual)



if __name__ == "__main__":
  train()