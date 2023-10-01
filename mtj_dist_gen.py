# ===== handles fortran interface =====
from interface_funcs import mtj_sample
# ===========================================================
from config_verify import config_verify
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mtj_types_v3 import SHE_MTJ_rng, VCMA_MTJ_rng, SWrite_MTJ_rng
from jz_lut import jz_lut_she, jz_lut_vcma, jz_lut_write

cdf      = lambda x, lmda: 1-np.exp(-lmda*x)
make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))

def dist_rng(dev,k,init,lmda,\
             dump_mod,mag_view_flag):
    x2 = 2**k
    x0 = 1
    x1 = (x2+x0)/2
    theta = init
    phi = np.random.rand()*2*np.pi
    dev.set_mag_vector(phi,theta)
    number = 0
    bits = []
    energies = []

    for i in range(k):
      pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
      # ===================== entry point to fortran interface ==========================
      out,energy = mtj_sample(dev,jz_lut_she(pright),dump_mod,mag_view_flag)
      #out,energy = mtj_sample(dev,jz_lut_vcma(pright),dump_mod,mag_view_flag)
      #out,energy = mtj_sample(dev,jz_lut_write(pright),dump_mod,mag_view_flag)
      # ==============================================================================================
      bits.append(out)
      energies.append(energy)

      if out == 1:
        x0 = x1
      elif out == 0:
        x2 = x1
      x1 = (x2+x0)/2
      number += out*2**(k-i-1)
    return number,bits,energies

def main():
  dev = SHE_MTJ_rng()
  dev.set_vals(0) # 1 uses default values with dev-to-dev variation on, 0, off
  print(dev)      # can print device to list all parameters
  print("verifying device paramters")
  nerr, mz1, mz2, PI = config_verify(dev)
  # ignoring warnings
  if nerr == -1:
    print('numerical error, do not use parameters!')
  elif PI == -1:
    print('PMA too strong')
  elif PI == 1:
    print('IMA too strong')
  else:
    print('parameters okay')
  print("running application")
  exit()

  k       = 8
  lmda    = 0.01
  init_t  = 9*np.pi/10
  samples = 8000
  number_history = []
  bitstream  = []
  energy_avg = []
  mag_view_flag = 1
  dump_mod      = 100 # dump phi and theta history every value (only applicable if mag_view_flag is true)

  for j in range(samples):
      number_j,bits_j,energies_j = dist_rng(dev,k,init_t,lmda,dump_mod,mag_view_flag)
      number_history.append(number_j)
      bitstream.append(''.join(str(i) for i in bits_j))
      energy_avg.append(np.average(energies_j))

  # Build an analytical exponential probability density function (PDF)
  xxis = []
  exp_pdf = []
  exp_count, _ = np.histogram(number_history,bins=256)
  for j in range(256):
    if (j == 0):
      xxis.append(j)
      exp_pdf.append(exp_count[0])

    if (j > 0):
      number = lmda*np.exp(-lmda*j)
      number = number*exp_pdf[0]/lmda
      xxis.append(j)
      exp_pdf.append(number)

  # Normalize exponential distribution
  expsum = 0
  for j in range(256):
    expsum += exp_pdf[j]

  exp_pdf = exp_pdf/expsum
  exp_pdf = exp_pdf*samples
  # get counts from number_history
  counts, _ = np.histogram(number_history,bins=256)

  # Calculate the chi_square parameter
  chi2 = 0
  for j in range(256):
    chi2 += ((counts[j]-exp_pdf[j])**2)/exp_pdf[j]


  # File holds the Chi2 value
  make_dir("results")
  chi2Data_path = "results/chi2Data.txt"
  f = open(chi2Data_path,'w')
  f.write(str(chi2))
  f.close()

  #normalize
  counts = counts/samples
  exp_pdf = exp_pdf/samples

  # Build plot 2, overlay computed distribution to ideal exponential
  plt.figure(2)
  plt.plot(xxis, counts, 'b-')
  plt.plot(xxis, exp_pdf,'k--')
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  distribution_plot_path = "results/distribution_plot.png"
  plt.savefig(distribution_plot_path)
  plt.clf()
  plt.close()

if __name__ == "__main__":
  main()
