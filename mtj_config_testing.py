# ===== handles fortran interface and batch parallelism =====
from interface_funcs import run_in_parallel_batch, mtj_sample
# ===========================================================
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
    #FIXME
    #print(s.getvalue())
    return retval
  return inner

# =====================================================================
#       NOTE: do not call this function directly —              
#       arguments handeled by interface function "run_in_parllel_batch"
# =====================================================================
'''
def single_run_parallel(dev,k,init,lmda,\
                      hist_queue,bitstream_queue,energy_avg_queue,mag_view_flag,proc_ID):
      temp,bits,energies = rng(dev,k,init,lmda,mag_view_flag,proc_ID)
      hist_queue.put(temp)
      bitstream_queue.put(''.join(str(i) for i in bits))
      energy_avg_queue.put(np.average(energies))
'''

# =====================================================================
#            to be called directly in serial operation
# =====================================================================
def single_run_serial(dev,k,init,lmda,\
                      hist,bitstream,energy_avg,mag_view_flag,iterator_never_equal_6):
      temp,bits,energies = rng(dev,k,init,lmda,mag_view_flag,iterator_never_equal_6)
      return temp,bits,energies

def single_run_parallel(dev,k,init,lmda,\
                      hist_queue,bitstream_queue,energy_avg_queue,mag_view_flag,proc_ID):
    x2 = 2**k
    x0 = 1
    x1 = (x2+x0)/2
    theta = init
    phi = np.random.rand()*2*np.pi
    # NOTE: new method
    dev.set_mag_vector(phi,theta)
    temp = 0
    bits = []
    energies = []

    for i in range(k):
      pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
      out,energy = mtj_sample(dev,jz_lut_she(pright),mag_view_flag,proc_ID)
      bits.append(out)
      energies.append(energy)

      if out == 1:
        x0 = x1
      elif out == 0:
        x2 = x1
      x1 = (x2+x0)/2
      temp += out*2**(k-i-1)

    hist_queue.put(temp)
    bitstream_queue.put(''.join(str(i) for i in bits))
    energy_avg_queue.put(np.average(energies))

def rng(dev,k,init,lmda,mag_view_flag,proc_ID):
    x2 = 2**k
    x0 = 1
    x1 = (x2+x0)/2
    theta = init
    phi = np.random.rand()*2*np.pi
    # NOTE: new method
    dev.set_mag_vector(phi,theta)
    temp = 0
    bits = []
    energies = []

    for i in range(k):
      pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
      out,energy = mtj_sample(dev,jz_lut_she(pright),mag_view_flag,proc_ID)
      bits.append(out)
      energies.append(energy)

      if out == 1:
        x0 = x1
      elif out == 0:
        x2 = x1
      x1 = (x2+x0)/2
      temp += out*2**(k-i-1)
    return temp,bits,energies

def cdf(x, lmda):
    return 1-np.exp(-lmda*x)


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
def mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, writeFile=None):
  # if writeFile == None:
  #   csvFile = "MTJ_Results.csv"
  #   f = open(csvFile, "a")
  #   writeFile = csv.writer(f)

  dd = 1
  # NOTE: device init only takes in dev-to-dev variation flag
  dev = SHE_MTJ_rng(dd_flag=dd)
  # NOTE: parameter setting done manually or with set_vals method
  dev.set_vals(1)
  #print(dev)

  k = 8
  lmda = 0.01
  init_t = 9*np.pi/10
  samples = 100
  hist = []
  bitstream = []
  energy_avg = []
  mag_view_flag = False
  # FIXME: implement safeguard to prevent computer bog-down???
  parallel_flag = True
  parallel_batch_size = None # ==== NOTE:  None value defaults to the total number of cores on the CPU ====

  # ===================== entry point to parallelized fortran interface ==========================
  # This function will spawn multiple processes of single_run_wrapper which then passes
  # the dynamical computation jobs to fortran. After this functions return, there are no
  # more diverges from this python script.
  # ========================================
  if (parallel_flag):
      hist, energy_avg, bitstream = run_in_parallel_batch(single_run_parallel,samples,\
                                                            dev,k,init_t,lmda,hist,bitstream,energy_avg,\
                                                            mag_view_flag,parallel_batch_size)
  else:
      for j in range(samples):
          temp_j,bits_j,energies_j = single_run_serial(dev,k,init_t,lmda,hist,bitstream,energy_avg,\
                                                             mag_view_flag,j+7)
          hist.append(temp_j)
          bitstream.append(''.join(str(i) for i in bits_j))
          energy_avg.append(np.average(energies_j))

  # ==============================================================================================
  # Build an analytical exponential probability density function (PDF)
  xxis = []
  exp_pdf = []
  exp_count, _ = np.histogram(hist,bins=256)
  for j in range(256):
    if (j == 0):
      temp = exp_count[0]
      xxis.append(j)
      exp_pdf.append(temp)
        
    if (j > 0):
      temp = lmda*np.exp(-lmda*j)
      temp = temp*exp_pdf[0]/lmda
      xxis.append(j)
      exp_pdf.append(temp)
  
  # Normalize exponential distribution
  expsum = 0
  for j in range(256):
    expsum += exp_pdf[j]
  
  exp_pdf = exp_pdf/expsum
  exp_pdf = exp_pdf*samples
  counts, _ = np.histogram(hist,bins=256)




  # Calculate the chi_square parameter
  chi2 = 0
  for j in range(256):
    chi2 += ((counts[j]-exp_pdf[j])**2)/exp_pdf[j]

  check_output_paths()
  # File holds the Chi2 value
  chi2Data_path = "results/chi2Data/chi2Data_{}.txt".format(run)
  f = open(chi2Data_path,'w')
  f.write(str(chi2))
  f.close
      
  counts = counts/samples
  exp_pdf = exp_pdf/samples

  # Build plot 2, overlay computed distribution to ideal exponential
  plt.figure(2)
  plt.plot(xxis, counts, 'b-')
  plt.plot(xxis, exp_pdf,'k--')
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  distribution_plot_path = "results/plots/distribution_plots/distribution_plot_{}.png".format(run)
  plt.savefig(distribution_plot_path)
  plt.clf()
  plt.close()

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

  fig, axs = plt.subplots(1,2)
  axs[0].scatter(yvals,zvals)
  axs[0].axis('equal')
  axs[0].axis(xmin = -1, xmax = 1)
  axs[0].axis(ymin = -1, ymax = 1)
  axs[0].set_xlabel('Y')
  axs[0].set_ylabel('Z')
  axs[1].scatter(xvals,zvals)
  axs[1].axis('equal')
  axs[1].axis(xmin = -1, xmax = 1)
  axs[1].axis(ymin = -1, ymax = 1)
  axs[1].set_xlabel('X')
  axs[1].set_ylabel('Z')
  axs[1].yaxis.set_label_position("right")
  axs[1].yaxis.tick_right()
  fig.tight_layout()
  magnetization_plot_path = "results/plots/magnetization_plots/magnetization_plot_{}.png".format(run)
  plt.savefig(magnetization_plot_path)
  plt.clf()
  plt.close()

  # Save bitstream and energy values
  bitstream_path = "results/bitstream_results/bitstream_{}.npy".format(run)
  np.save(bitstream_path, np.array(bitstream))
  energy_path = "results/energy_results/energy_{}.npy".format(run)
  np.save(energy_path, np.array(energy_avg))

  # File holds the number of times each number was generated; use for a histogram
  countData_path = "results/countData/countData_{}.txt".format(run)
  f = open(countData_path,'w')
  for i in range(256):
    f.write(str(counts[i]))
    f.write('\n')
  f.close

  # File holds the list of all random numbers generated at each sample
  bitData_path = "results/bitData/bitData_{}.txt".format(run)
  f = open(bitData_path,'w')
  for i in range(samples):
    f.write(str(hist[i]))
    f.write('\n')
  f.close

  # File holds the outputs of the magnetization path generating the first random number (only theta)
  magTheta_path = "results/magTheta/magTheta_{}.txt".format(run)
  f = open(magTheta_path,'w')
  try:
    for j in range(1):
      for i in range(500): 
        f.write(str(dev.thetaHistory[j][i]))
        f.write('\n')
  except IndexError:
    pass
  finally:
    f.close()

  # File holds the outputs of the magnetization path generating the first random number (only phi)
  magPhi_path = "results/magPhi/magPhi_{}.txt".format(run)
  f = open(magPhi_path,'w')
  try:
    for j in range(1):
      for i in range(500):
        f.write(str(dev.phiHistory[j][i]))
        f.write('\n')
  except IndexError:
    pass
  finally:
    f.close

  if writeFile == None:
    parameterFile_path = "results/parameter_files/parameterFile_{}.txt".format(run)
    with open(parameterFile_path, "w") as f:
      f.write("alpha: {}\n".format(str(alpha)))
      f.write("Ki: {}\n".format(str(Ki)))
      f.write("Ms: {}\n".format(str(Ms)))
      f.write("Rp: {}\n".format(str(Rp)))
      f.write("TMR: {}\n".format(str(TMR)))
      f.write("d: {}\n".format(str(d)))
      f.write("tf: {}\n".format(str(tf)))
      f.write("eta: {}\n".format(str(eta)))
      f.write("J_she: {}\n".format(str(J_she)))
      f.write("distribution_plot_path: {}\n".format(distribution_plot_path))
      f.write("magnetization_plot_path: {}\n".format(magnetization_plot_path))
      f.write("bitstream_path: {}\n".format(bitstream_path))
      f.write("energy_path: {}\n".format(energy_path))
      f.write("countData_path: {}\n".format(countData_path))
      f.write("bitData_path: {}\n".format(bitData_path))
      f.write("magTheta_path: {}\n".format(magTheta_path))
      f.write("magPhi_path: {}\n".format(magPhi_path))
      f.write("chi2Data_path: {}\n".format(chi2Data_path))
  else:
    writeFile.writerow([alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, 
                        distribution_plot_path, magnetization_plot_path, bitstream_path, 
                        energy_path, countData_path, bitData_path, magTheta_path, magPhi_path, chi2Data_path])


def main():
  #alpha_vals   = [0.01, 0.03, 0.05, 0.07, 0.1]            # damping constant
  #Ki_vals      = [0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1e-3]   # anistrophy energy    
  #Ms_vals      = [0.3e6, 0.7e6, 1.2e6, 1.6e6, 2e6]        # saturation magnetization
  #Rp_vals      = [500, 1000, 5000, 25000, 50000]          # parallel resistance
  #TMR_vals     = [0.3, 0.5, 2, 4, 6]                      # tunneling magnetoresistance ratio
  #d_vals       = [50]                                     # free layer diameter
  #tf_vals      = [1.1]                                    # free layer thickness
  #eta_vals     = [0.1, 0.2, 0.4, 0.6, 0.8]                # spin hall angle
  #J_she_vals   = [0.01e12, 0.1e12, 0.25e12, 0.5e12, 1e12] # current density

  alpha_vals   = [0.01]            # damping constant
  Ki_vals      = [0.2e-3]   # anistrophy energy    
  Ms_vals      = [0.3e6]        # saturation magnetization
  Rp_vals      = [500]          # parallel resistance
  TMR_vals     = [0.3]                      # tunneling magnetoresistance ratio
  d_vals       = [50]                                     # free layer diameter
  tf_vals      = [1.1]                                    # free layer thickness
  eta_vals     = [0.1]                # spin hall angle
  J_she_vals   = [0.01e12] # current density

  csvFile = "MTJ_Results.csv"
  f = open(csvFile, "w")
  writeFile = csv.writer(f)
  writeFile.writerow(['alpha', 'Ki', 'Ms', 'Rp', 'TMR', 'd', 'tf', 'eta', 'J_she', 
                      'distribution_plot_path', 'magnetization_plot_path', 'bitstream_path', 'energy_path', 'countData_path', 'bitData_path', 'magTheta_path', 'magPhi_path', 'chi2Data_path'])

  total_num_runs = len(alpha_vals)*len(Ki_vals)*len(Ms_vals)*len(Rp_vals)*\
               len(TMR_vals)*len(d_vals)*len(tf_vals)*len(eta_vals)*len(J_she_vals)
  pbar = tqdm(total=total_num_runs,ncols=80)
  run = 0
  for alpha in alpha_vals:
    for Ki in Ki_vals:
      for Ms in Ms_vals:
        for Rp in Rp_vals:
          for TMR in TMR_vals:
            for d in d_vals:
              for tf in tf_vals:
                for eta in eta_vals:
                    for J_she in J_she_vals:
                      mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, writeFile=writeFile)
                      run += 1
  pbar.update(1)

  f.close()

if __name__ == "__main__":
  start_time = time.time()
  main()
  print("--- %s seconds ---" % (time.time() - start_time))
  exit()

  # parser = argparse.ArgumentParser(description="MTJ Parameter Testing")
  # parser.add_argument("--alpha", required=True, help="damping constant", type=float)
  # parser.add_argument("--Ki", required=True, help="anistrophy energy", type=float)
  # parser.add_argument("--Ms", required=True, help="saturation magnetization", type=float)
  # parser.add_argument("--Rp", required=True, help="parallel resistance", type=float)
  # parser.add_argument("--TMR", required=True, help="tunneling magnetoresistance ratio", type=float)
  # parser.add_argument("--d", required=True, help="free layer diameter", type=float)
  # parser.add_argument("--tf", required=True, help="free layer thickness", type=float)
  # parser.add_argument("--eta", required=True, help="spin hall angle", type=float)
  # parser.add_argument("--J_she", required=True, help="current density", type=float)
  # parser.add_argument("--run", required=True, help="run number", type=int)
  # args = parser.parse_args()
  
  # alpha = args.alpha
  # Ki = args.Ki
  # Ms = args.Ms
  # Rp = args.Rp
  # TMR = args.TMR
  # d = args.d
  # tf = args.tf
  # eta = args.eta
  # J_she = args.J_she
  # run = args.run

  # print(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run)
  # mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, writeFile=None)

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

  # print(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run)
  mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, writeFile=None)
  
    
