import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import rel_entr

import sys
sys.path.append("../")
sys.path.append("../fortran_source")
from mtj_model import mtj_run

SAMPLES = 100000


if __name__ == "__main__":
  
  #TODO: Change param vals
  # alpha = 0.01
  # Ki = 0.0002
  # Ms = 300000.0
  # Rp = 500.0
  # TMR = 0.3
  # eta = 0.26508163213729863
  # J_she = 1000000000000.0
  # t_pulse = 3.790034800767898e-08
  # t_relax = 3.790034800767898e-08
  # d = 3e-09
  # tf = 1.1e-09

  alpha = 0.01
  Ki = 0.0005543247938156128
  Ms = 300000.0
  Rp = 500.0
  TMR = 0.3
  eta = 0.41814986765384676
  J_she = 426899464726.44806
  t_pulse = 5e-10
  t_relax = 5e-10
  d = 3e-09
  tf = 1.1e-09
  
  _, _, _, countData, _, xxis, exp_pdf = mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=SAMPLES)

  plt.plot(xxis, countData, color="red", label="Actual PDF")
  plt.plot(xxis, exp_pdf,'k--', label="Expected PDF")
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  plt.title("PDF Comparison")
  plt.legend()
  plt.show()