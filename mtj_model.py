# ===== handles fortran interface and batch parallelism =====
from interface_funcs import mtj_sample
# ===========================================================
from config_verify import config_verify
import os
import csv
import time
import argparse
import cProfile, pstats, io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtj_types_v3 import SHE_MTJ_rng
from jz_lut import jz_lut_she#,jz_lut_vcma,jz_lut_write

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
    #print(s.getvalue())
    return retval
  return inner

def dist_rng(dev,k,init,lmda,dump_mod_val,mag_view_flag,file_ID):
  x2 = 2**k
  x0 = 1
  x1 = (x2+x0)/2
  theta = init
  phi = np.random.rand()*2*np.pi
  dev.set_mag_vector(phi,theta)
  number = 0
  bits   = []
  energies = []

  for i in range(k):
    pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
    # ===================== entry point to fortran interface ==========================
    out,energy = mtj_sample(dev,jz_lut_she(pright),dump_mod_val,mag_view_flag,file_ID)
    bits.append(out)
    energies.append(energy)

    if out == 1:
      x0 = x1
    elif out == 0:
      x2 = x1
    x1 = (x2+x0)/2
    number += out*2**(k-i-1)
  return number,bits,energies

cdf       = lambda x,lmda: 1-np.exp(-lmda*x)
dir_check = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))

def check_output_paths() -> None:
  dir_check("./results")
  dir_check("./results/parameter_files")
  dir_check("./results/chi2Data")
  dir_check("./results/plots")
  dir_check("./results/plots/distribution_plots")
  dir_check("./results/plots/magnetization_plots")
  dir_check("./results/magPhi")
  dir_check("./results/bitstream_results")
  dir_check("./results/magTheta")
  dir_check("./results/energy_results")
  dir_check("./results/countData")
  dir_check("./results/bitData")

@profile
def mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=1000):
  dev = SHE_MTJ_rng()
  dev.set_vals(1) # only using default a and b, overwriting with below 
  dev.alpha = alpha
  dev.Ki = Ki
  dev.Ms = Ms
  dev.Rp = Rp
  dev.TMR = TMR
  dev.d = d
  dev.tf = tf
  dev.eta = eta
  dev.J_she = J_she
  dev.t_pulse = t_pulse
  dev.t_relax = t_relax
  # print(dev)
  
  # Verifying device paramters
  nerr, mz1, mz2, PI = config_verify(dev, runID=0)
  if nerr == -1:
    # print('numerical error, do not use parameters!')
    return None, None, None, None, None, None, None
  elif PI == -1:
    # print('PMA too strong')
    return None, None, None, None, None, None, None
  elif PI == 1:
    # print('IMA too strong')
    return None, None, None, None, None, None, None
  else:
    pass

  k       = 8
  lmda    = 0.01
  init_t  = 9*np.pi/10
  # samples = 100000
  number_history = []
  bitstream  = []
  energy_avg = []
  mag_view_flag = False
  dump_mod_val  = 8000

  for j in range(samples):
    number_j,bits_j,energies_j = dist_rng(dev,k,init_t,lmda,dump_mod_val,mag_view_flag,j+7)
    number_history.append(number_j)
    bitstream.append(''.join(str(i) for i in bits_j))
    energy_avg.append(np.average(energies_j))


  # ==============================================================================================
  # Build an analytical exponential probability density function (PDF)
  xxis = []
  exp_pdf = []
  # exp_count, _ = np.histogram(number_history,bins=256)
  for j in range(256):
    if (j == 0):
      xxis.append(j)
      temp = 1
      exp_pdf.append(temp)

    if (j > 0):
      number = lmda*np.exp(-lmda*j)
      number = number*exp_pdf[0]/lmda
      xxis.append(j)
      exp_pdf.append(number)

  # Normalize exponential distribution
  expsum = 0
  for j in range(256):
    expsum += exp_pdf[j]

  # print("expsum:", expsum)
  exp_pdf = exp_pdf/expsum
  exp_pdf = exp_pdf*samples
  counts, _ = np.histogram(number_history,bins=256)


  # Calculate the chi_square parameter
  chi2 = 0
  for j in range(256):
    chi2 += ((counts[j]-exp_pdf[j])**2)/exp_pdf[j]

  # check_output_paths()

  counts = counts/samples
  exp_pdf = exp_pdf/samples

  # Build plot 3, will track the magnetization path of the first generated random number
  # look at this picture to see if IMA or PMA; if magnetization stays strongly in XY plane,
  # it is IMA. If magnetization never comes down from +Z axis, PMA is too strong
  # if simulation durations change, inner for loop (for in in range (500) will need to change
  xvals = []
  yvals = []
  zvals = []

  cnt = 0
  try:
    for j in range(1):
      for i in range(500): # change range
        xvals.append(np.sin(dev.thetaHistory[j][i])*np.cos(dev.phiHistory[j][i]))
        yvals.append(np.sin(dev.thetaHistory[j][i])*np.sin(dev.phiHistory[j][i]))
        zvals.append(np.cos(dev.thetaHistory[j][i]))
        cnt += 1
  except IndexError:
    pass

  return chi2, bitstream, energy_avg, counts[0:256], number_history[0:samples], xxis, exp_pdf


def main():
  parser = argparse.ArgumentParser(description="MTJ Parameter Testing")
  parser.add_argument("--ID", required=True, help="run ID", type=int)
  args = parser.parse_args()
  ID = args.ID

  f = np.load("parameter_config.npy")
  
  run = ID
  # run = ID + 19999
  alpha = f[run][0]
  Ki = f[run][1]
  Ms = f[run][2]
  Rp = f[run][3]
  TMR = f[run][4]
  d = f[run][5]
  tf = f[run][6]
  eta = f[run][7]
  J_she = f[run][8]
  t_pulse = f[run][9]
  t_relax = f[run][10]

  mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=1000)


if __name__ == "__main__":
  start_time = time.time()
  main()
  print("--- %s seconds ---" % (time.time() - start_time))
  exit()
