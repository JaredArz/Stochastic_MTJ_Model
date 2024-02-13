import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

from mtj_types  import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
from mtj_helper import valid_config, get_energy, gamma_pdf
from interface_funcs import mtj_sample, mtj_check
from jz_lut import jz_lut_she



def SOT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, eta, J_reset, t_pulse, t_relax, samples=1000):
  dev = SWrite_MTJ_rng(flavor="UTA")
  dev.init() # calls both set_vals and set_mag_vector with defaultsdev.alpha = alpha
  # print(dev)
  # exit()
  dev.set_vals(alpha=alpha,
               K_295=K_295,
               Ms_295=Ms_295,
               Rp=Rp,
               TMR=TMR,
               d=d,
               tf=tf,
               eta=eta,
               J_reset=J_reset,
               t_pulse=t_pulse,
               t_relax=t_relax)

  # Check if config is valid
  valid = valid_config(*mtj_check(dev, 0, 100))
  if valid == False:
    return None, None, None, None, None, None, None

  # Sample device to get bitstream and energy consumption
  number_history, bitstream, energy_avg = get_energy(dev, samples, jz_lut_she) #TODO: Change jz_lut_she!!!!
  
  # Build gamma distribution
  # xxis, pdf = gamma_pdf(g1=1, g2=0.01, nrange=256)
  xxis, pdf = gamma_pdf(g1=50, g2=311.44, nrange=256)
  
  # Calculate chi2
  counts, _ = np.histogram(number_history,bins=256)
  pdf = pdf*samples
  chi2 = 0
  for j in range(256):
    chi2 += ((counts[j]-pdf[j])**2)/pdf[j]

  counts = counts/samples
  pdf = pdf/samples

  return chi2, bitstream, energy_avg, counts[0:256], number_history[0:samples], xxis, pdf



if __name__ == "__main__":
  alpha = 0.01
  K_295 = 0.0002
  Ms_295 = 300000
  Rp = 13265.555784106255
  TMR = 0.3
  eta = 0.8
  J_reset = 334994280934.3338
  t_pulse = 7.5e-08
  t_relax = 7.5e-08
  d = 3e-09
  tf = 1.1e-09
  SAMPLES = 100000

  chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = SOT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, eta, J_reset, t_pulse, t_relax, samples=SAMPLES)
  kl_div_score = sum(rel_entr(countData, exp_pdf))
  energy = np.mean(energy_avg)

  print("KL_Div:", kl_div_score)
  print("Energy:", energy)
  print()

  plt.plot(xxis, countData, color="red", label="Actual PDF")
  plt.plot(xxis, exp_pdf,'k--', label="Expected PDF")
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  plt.title("PDF Comparison")
  plt.legend()
  plt.show()



  '''
  alpha_range = [0.01, 0.1]
  Ki_range = [0.2e-3, 1e-3]
  Ms_range = [0.3e6, 2e6]
  Rp_range = [500, 50000]
  TMR_range = [0.3, 6] # No longer vary; default to 3
  eta_range = [0.1, 2] # Only for SOT
  J_she_range = [0.01e12, 5e12] # Only for SOT
  J_stt_range = [] # Only for STT
  J_reset = [] # Only for STT (3 * J_stt)
  t_reset = [] # Only for STT (3 * t_pulse)
  t_pulse_range = [0.5e-9, 75e-9]
  t_relax_range = [0.5e-9, 75e-9]
  '''