import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

from mtj_types  import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
from mtj_helper import dev_check, get_energy, get_pdf
from interface_funcs import mtj_sample, mtj_check
from jz_lut import jz_lut_write



# def STT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, eta, J_stt, t_pulse, t_relax, samples=1000):
#   dev = SWrite_MTJ_rng(flavor="UTA")
#   dev.init() # calls both set_vals and set_mag_vector with defaultsdev.alpha = alpha

#   J_reset = 3*J_stt
#   t_reset = 3*t_pulse

#   dev.set_vals(alpha=alpha,
#                K_295=K_295,
#                Ms_295=Ms_295,
#                Rp=Rp,
#                TMR=TMR,
#                d=d,
#                tf=tf,
#                eta=eta,
#                J_reset=J_reset,
#                t_reset=t_reset,
#                t_pulse=t_pulse,
#                t_relax=t_relax)
  

#   # Check if config is valid
#   valid = valid_config(*mtj_check(dev, J_stt, 100))
#   if valid == False:
#     return None, None, None, None, None, None, None

#   # Build gamma distribution
#   pdf_type = "exp"
#   # pdf_type = "gamma"
#   xxis, pdf = get_pdf(pdf_type)

#   # Sample device to get bitstream and energy consumption
#   number_history, bitstream, energy_avg = get_energy(dev, samples, jz_lut_write, pdf_type)
  
#   # Calculate chi2
#   counts, _ = np.histogram(number_history,bins=256)
#   pdf = pdf*samples
#   chi2 = 0
#   for j in range(256):
#     chi2 += ((counts[j]-pdf[j])**2)/pdf[j]

#   counts = counts/samples
#   pdf = pdf/samples

#   return chi2, bitstream, energy_avg, counts[0:256], number_history[0:samples], xxis, pdf


def STT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, t_pulse, t_relax, samples=1000, pdf_type="exp"):
  dev = SWrite_MTJ_rng(flavor="UTA")
  dev.init() # calls both set_vals and set_mag_vector with defaults
  dev.set_vals(alpha=alpha,
               K_295=K_295,
               Ms_295=Ms_295,
               Rp=Rp,
               TMR=TMR,
               d=d,
               tf=tf,
               t_reset=3*t_pulse,
               t_pulse=t_pulse,
               t_relax=t_relax)
  
  # Check if config is valid
  valid, scurve = dev_check(dev, plot=False)
  if valid == False:
    return None, None, None, None, None, None, None

  # Build gamma distribution
  xxis, pdf = get_pdf(pdf_type)

  # Sample device to get bitstream and energy consumption
  number_history, bitstream, energy_avg = get_energy(dev, samples, scurve, pdf_type)
  
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
  SAMPLES = 2500
  # PDF_TYPE = "exp"
  PDF_TYPE = "gamma"

  alpha = 0.03
  K_295 = 1.0056364e-3
  Ms_295 = 1.2e6
  Rp = 5e3
  TMR = 3
  t_pulse = 1e-9
  t_relax = 10e-9
  d = 3e-09
  tf = 1.1e-09

  chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, t_pulse, t_relax, samples=SAMPLES, pdf_type=PDF_TYPE)
  kl_div_score = sum(rel_entr(countData, pdf))
  energy = np.mean(energy_avg)
  
  print("Chi2  :", chi2)
  print("KL_Div:", kl_div_score)
  print("Energy:", energy)

  plt.plot(xxis, countData, color="red", label="Actual PDF")
  plt.plot(xxis, pdf,'k--', label="Expected PDF")
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
  J_stt_range = [-136090192630.6827, -113236607131.10739] # Only for STT
  J_reset = [] # Only for STT (3 * J_stt)
  t_reset = [] # Only for STT (3 * t_pulse)
  t_pulse_range = [0.5e-9, 75e-9]
  t_relax_range = [0.5e-9, 75e-9]
  '''