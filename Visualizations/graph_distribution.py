import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import rel_entr

import sys
sys.path.append("../")
sys.path.append("../fortran_source")
from SOT_model import SOT_Model
from STT_model import STT_Model


def graph_SOT(params, pdf_type, samples):
  alpha = params["alpha"]
  Ki = params["Ki"]
  Ms = params["Ms"]
  Rp = params["Rp"]
  TMR = params["TMR"]
  eta = params["eta"]
  J_she = params["J_she"]
  t_pulse = params["t_pulse"]
  t_relax = params["t_relax"]
  d = params["d"]
  tf = params["tf"]
  
  chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = SOT_Model(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples, pdf_type)
  
  if chi2 == None:
    raise Exception("Configuration checks failed")

  kl_div_score = sum(rel_entr(countData, pdf))
  energy = np.mean(energy_avg)
  
  print("Chi2  :", chi2)
  print("KL_Div:", kl_div_score)
  print("Energy:", energy)
  
  plt.plot(xxis, countData, color="red", label="Actual PDF")
  plt.plot(xxis, pdf,'k--', label="Expected PDF")
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  plt.title(f"SOT {pdf_type.capitalize()} PDF Comparison")
  plt.legend()
  plt.show()


def graph_STT(params, pdf_type, samples):
  alpha = params["alpha"]
  K_295 = params["K_295"]
  Ms_295 = params["Ms_295"]
  Rp = params["Rp"]
  TMR = params["TMR"]
  t_pulse = params["t_pulse"]
  t_relax = params["t_relax"]
  d = params["d"]
  tf = params["tf"]

  chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(alpha, K_295, Ms_295, Rp, TMR, d, tf, t_pulse, t_relax, samples, pdf_type)
  
  if chi2 == None:
    raise Exception("Configuration checks failed")
  
  kl_div_score = sum(rel_entr(countData, pdf))
  energy = np.mean(energy_avg)
  
  print("Chi2  :", chi2)
  print("KL_Div:", kl_div_score)
  print("Energy:", energy)

  plt.plot(xxis, countData, color="red", label="Actual PDF")
  plt.plot(xxis, pdf,'k--', label="Expected PDF")
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  plt.title(f"STT {pdf_type.capitalize()} PDF Comparison")
  plt.legend()
  plt.show()



if __name__ == "__main__":
  samples = 100_000
  pdf_type = "gamma"

  SOT_params = {
    "alpha"   : 0.03,
    "Ki"      : 1.0056364e-3,
    "Ms"      : 1.2e6,
    "Rp"      : 5e3,
    "TMR"     : 3,
    "eta"     : 0.3,
    "J_she"   : 5e11,
    "t_pulse" : 10e-9,
    "t_relax" : 15e-9,
    "d"       : 3e-09,
    "tf"      : 1.1e-09
  }

  STT_params = {
    "alpha"   : 0.03,
    "K_295"   : 1.0056364e-3,
    "Ms_295"  : 1.2e6,
    "Rp"      : 5e3,
    "TMR"     : 3,
    "t_pulse" : 1e-9,
    "t_relax" : 10e-9,
    "d"       : 3e-09,
    "tf"      : 1.1e-09
  }

  graph_SOT(SOT_params, pdf_type, samples)
  # graph_STT(STT_params, pdf_type, samples)