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
  alpha = 1.00000000e-01
  Ki = 5.68017578e-04
  Ms = 9.24002456e+05
  Rp = 1.79756189e+04
  TMR = 5.82181854e-01
  eta = 6.60312960e-01
  J_she = 3.18389271e+11
  t_pulse = 7.50000000e-08
  t_relax = 7.50000000e-08
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