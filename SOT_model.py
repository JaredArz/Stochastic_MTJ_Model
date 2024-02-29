import os
import sys
import random
import numpy as np
import scipy.stats as stats 
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import cProfile, pstats, io

from mtj_types  import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
# from mtj_helper import valid_config, get_energy, gamma_pdf, get_pdf
from mtj_helper2 import dev_check, get_energy, get_pdf
from interface_funcs import mtj_sample, mtj_check
from jz_lut import jz_lut_she



def profile(func):
  def inner(*args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    retval = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #NOTE: diagnostics
    # print(s.getvalue())
    return retval
  return inner

# def SOT_Model(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=1000):
#   dev = SHE_MTJ_rng()
#   dev.init() # calls both set_vals and set_mag_vector with defaultsdev.alpha = alpha
#   dev.set_vals(alpha=alpha,
#                Ki=Ki,
#                Ms=Ms,
#                Rp=Rp,
#                TMR=TMR,
#                d=d,
#                tf=tf,
#                eta=eta,
#                J_she=J_she,
#                t_pulse=t_pulse,
#                t_relax=t_relax)

#   # Check if config is valid
#   valid = valid_config(*mtj_check(dev, 0, 100))
#   if valid == False:
#     return None, None, None, None, None, None, None
  
#   # Build gamma distribution
#   pdf_type = "exp"
#   xxis, pdf = get_pdf(pdf_type)

#   # Sample device to get bitstream and energy consumption
#   number_history, bitstream, energy_avg = get_energy(dev, samples, jz_lut_she, pdf_type)
  
#   # Calculate chi2
#   counts, _ = np.histogram(number_history, bins=256)
#   pdf = pdf*samples
#   chi2 = 0
#   for j in range(256):
#     chi2 += ((counts[j]-pdf[j])**2)/pdf[j]

#   counts = counts/samples
#   pdf = pdf/samples

#   return chi2, bitstream, energy_avg, counts[0:256], number_history[0:samples], xxis, pdf


@profile
def SOT_Model(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=1000, pdf_type="exp"):
  dev = SHE_MTJ_rng()
  dev.init() # calls both set_vals and set_mag_vector with defaults
  dev.set_vals(alpha=alpha,
               Ki=Ki,
               Ms=Ms,
               Rp=Rp,
               TMR=TMR,
               d=d,
               tf=tf,
               eta=eta,
               J_she=J_she,
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
  counts, _ = np.histogram(number_history, bins=256)
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
  
  alpha = 0.01
  Ki = 0.0002
  Ms = 300000
  Rp = 13265.555784106255
  TMR = 0.3
  eta = 0.8
  J_she = 334994280934.3338
  t_pulse = 7.5e-08
  t_relax = 7.5e-08
  d = 3e-09
  tf = 1.1e-09
  
  chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = SOT_Model(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=SAMPLES, pdf_type=PDF_TYPE)
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