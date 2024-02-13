import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import rel_entr

import sys
sys.path.append("../")
sys.path.append("../fortran_source")
from SOT_model import SOT_Model



if __name__ == "__main__":
  SAMPLES = 100000

  # alpha = 0.01
  # Ki = 0.0002
  # Ms = 300000
  # Rp = 13265.555784106255
  # TMR = 0.3
  # eta = 0.8
  # J_she = 334994280934.3338
  # t_pulse = 7.5e-08
  # t_relax = 7.5e-08
  # d = 3e-09
  # tf = 1.1e-09

  alpha = 6.89516238e-02
  Ki = 2e-04
  Ms = 5.18540813e+05
  Rp = 1.59064407e+04
  TMR = 3
  eta = 2.25107130e-01
  J_she = 1.85007366e+12
  t_pulse = 5.0e-10
  t_relax = 5.0e-10
  d = 3e-09
  tf = 1.1e-09
  
  chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = SOT_Model(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=SAMPLES)
  kl_div_score = sum(rel_entr(countData, exp_pdf))
  energy = np.mean(energy_avg)

  print("KL_Div:", kl_div_score)
  print("Energy:", energy)

  plt.plot(xxis, countData, color="red", label="Actual PDF")
  plt.plot(xxis, exp_pdf,'k--', label="Expected PDF")
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  plt.title("PDF Comparison")
  plt.legend()
  plt.show()