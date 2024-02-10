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

  alpha = 0.01
  Ki = 0.0008099494457244873
  Ms = 828945.0943470001
  Rp = 10624.170124530792
  TMR = 2.0543512523174288
  eta = 0.8
  J_she = 577729867696.7621
  t_pulse = 1.3181074909865856e-08
  t_relax = 1.3181074909865856e-08
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