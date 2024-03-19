import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate
from scipy.signal import find_peaks
from scipy.special import rel_entr

from interface_funcs import mtj_sample
from mtj_types  import SWrite_MTJ_rng, SHE_MTJ_rng, VCMA_MTJ_rng


# GAMMA_LOWER = 0.05
# GAMMA_UPPER = 0.3
GAMMA_LOWER = 0.1
GAMMA_UPPER = 0.24

class S_Curve:
  def __init__(self, dev, x, num_to_avg):
    self.x = x
    self.y = self.generate(dev, num_to_avg=num_to_avg)

  def generate(self, dev, num_to_avg):
    weights = []
    for J in self.x:
      weights.append(self.avg_weight_across_samples(dev, J, num_to_avg))
    return weights
  
  def avg_weight_across_samples(self, dev, apply, samples_to_avg) -> float:
    # STT device does not need to be reset on sample
    if dev.mtj_type == 1:
      sum_p = np.sum( [ (mtj_sample(dev, apply),) for _ in range(samples_to_avg)] )
    else:
      sum_p = 0
      for _ in range(samples_to_avg):
        dev.set_mag_vector()
        bit,_ = mtj_sample(dev, apply)
        sum_p += bit
    return sum_p/samples_to_avg


def is_monotonic(x, y):
  peak_indices = find_peaks(y, prominence=0.1)[0]
  peak_x = []
  peak_y=[]
  for idx in peak_indices:
    peak_x.append(x[idx])
    peak_y.append(y[idx])
  
  monotonicity = True if len(peak_indices) == 0 else False
  return monotonicity, peak_x, peak_y


def dev_check(dev, plot=False):
  resolution = 50

  if type(dev) == SHE_MTJ_rng:
    current = np.linspace(-1e10, 1e10, resolution)
    title = "SHE"
  elif type(dev) == SWrite_MTJ_rng:
    current = np.linspace(-1e12, 0, resolution)
    title = "SWrite"
  elif type(dev) == VCMA_MTJ_rng:
    current = np.linspace(-1e10, 1e10, resolution)
    title = "VCMA"
  else:
    raise KeyError("Invalid dev type")
  
  scurve1 = S_Curve(dev, current, num_to_avg=500)
  
  delta = 0.025
  max_val = np.max(scurve1.y)
  min_val = np.min(scurve1.y)
  stop_flag = False
  start = 0
  end = len(scurve1.y)-1
  
  for i, val in enumerate(scurve1.y):
    if (val < max_val) and (max_val-val >= delta) and (stop_flag == False):
      start = i
      stop_flag = True
    if (val > min_val) and (val-min_val >= delta):
      end = i
  
  current2 = np.linspace(scurve1.x[start], scurve1.x[end], resolution)
  scurve2 = S_Curve(dev, current2, num_to_avg=750)
  monotonicity, peak_x, peak_y = is_monotonic(scurve2.x, scurve2.y)
  
  prob_diff = max(scurve2.y) - min(scurve2.y)
  validity = True if (monotonicity and prob_diff >= 0.8) else False
  
  if plot:
    plt.subplot(1, 2, 1)
    plt.scatter(scurve1.x, scurve1.y, s=36, marker="^", c="lightblue")
    plt.scatter(scurve1.x[start], scurve1.y[start], s=36, marker="^", c="green")
    plt.scatter(scurve1.x[end], scurve1.y[end], s=36, marker="^", c="red")
    plt.plot(scurve1.x, scurve1.y)
    plt.xlabel("J [A/m^2]")
    plt.title(f"S-Curve 1: {title}")

    plt.subplot(1, 2, 2)
    plt.scatter(scurve2.x, scurve2.y, s=36, marker="^", c="lightblue")
    plt.plot(scurve2.x, scurve2.y)
    plt.scatter(peak_x, peak_y, s=36, marker="^", c="red")
    plt.xlabel("J [A/m^2]")
    plt.title(f"S-Curve 2: {title}")
    plt.show()

  return validity, scurve2


def remap(num, inMin, inMax, outMin, outMax):
  rv = outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))
  return int(rv)


def get_pdf(type="exp"):
  if type == "exp":
    xxis = np.linspace(0, 255, 256)
    pdf = stats.gamma.pdf(xxis, a=1, scale=1/0.01)
    pdf = pdf/np.sum(pdf)
  elif type == "gamma":
    # xxis = np.linspace(0.05, 0.3, 256)
    xxis = np.linspace(GAMMA_LOWER, GAMMA_UPPER, 256)
    pdf = stats.gamma.pdf(xxis, a=50, scale=1/311.44)
    pdf = pdf/np.sum(pdf)
  else:
    raise TypeError("Invalid pdf type")
  
  return xxis, pdf


def dist_rng(dev, k, init, lmda, dump_mod_val, mag_view_flag, file_ID, scurve, pdf_type="exp"):
  if pdf_type == "exp":
    x2 = 2**k
    x0 = 1
    x1 = (x2+x0)/2
    cdf = lambda x,lmda: 1-np.exp(-lmda*x)
  elif pdf_type == "gamma":
    # x2 = 0.3
    # x0 = 0.05
    x2 = GAMMA_UPPER
    x0 = GAMMA_LOWER
    x1 = (x2+x0)/2
    xxis = np.linspace(x0, x2, 256)
    cdf_arr = stats.gamma.cdf(xxis, a=50, scale=1/311.44)
    # cdf = lambda x,lmda: cdf_arr[round(remap(x, 0.05, 0.3, 0, 255))]
    cdf = lambda x,lmda: cdf_arr[round(remap(x, GAMMA_LOWER, GAMMA_UPPER, 0, 255))]

  theta = init
  phi = np.random.rand()*2*np.pi
  dev.set_mag_vector(phi,theta)
  number = 0
  bits = []
  energies = []

  for i in range(k):
    pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
    if pright > max(scurve.y) or pright < min(scurve.y):
      pright = min(scurve.y, key=lambda x:abs(x-pright))
    f = interpolate.interp1d(scurve.y, scurve.x)
    current = f(pright)
    
    out,energy = mtj_sample(dev, current, mag_view_flag, dump_mod_val, file_ID)
    bits.append(out)
    energies.append(energy)

    if out == 1:
      x0 = x1
    elif out == 0:
      x2 = x1
    
    x1 = (x2+x0)/2
    number += out*2**(k-i-1)
  
  return number,bits,energies


def get_energy(dev, samples, scurve, pdf_type="exp"):
  k = 8
  lmda = 0.01
  init_t = 9*np.pi/10
  number_history = []
  bitstream = []
  energy_avg = []
  mag_view_flag = False
  dump_mod_val  = 8000

  for j in range(samples):
    number_j,bits_j,energies_j = dist_rng(dev, k, init_t, lmda, dump_mod_val, mag_view_flag, j+7, scurve, pdf_type)
    number_history.append(number_j)
    bitstream.append(''.join(str(i) for i in bits_j))
    energy_avg.append(np.average(energies_j))
  
  return number_history, bitstream, energy_avg


if __name__ == "__main__":
  # dev = SWrite_MTJ_rng("UTA")
  dev = SHE_MTJ_rng()
  dev.init()

  if type(dev) == SHE_MTJ_rng:
    # SOT Parameters
    samples = 1000
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
    
    # dev.set_vals(Ms=1.0*dev.Ms, t_pulse=1*dev.t_pulse)
    # dev.set_vals(Ms=1.0*dev.Ms, t_pulse=5*dev.t_pulse)
    # dev.set_vals(Ms=0.1*dev.Ms, t_pulse=1*dev.t_pulse)
    # dev.set_vals(Ms=0.1*dev.Ms, t_pulse=5*dev.t_pulse)

  elif type(dev) == SWrite_MTJ_rng:
    # STT Parameters
    alpha = 0.03
    K_295 = 1.0056364e-3
    Ms_295 = 1.2e6
    Rp = 5e3
    TMR = 3
    t_pulse = 1e-9
    t_relax = 10e-9
    d = 3e-09
    tf = 1.1e-09

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
    
    # dev.set_vals(Ms_295=1.0*dev.Ms_295, t_pulse=1*dev.t_pulse)
    # dev.set_vals(Ms_295=1.0*dev.Ms_295, t_pulse=5*dev.t_pulse)
    # dev.set_vals(Ms_295=0.1*dev.Ms_295, t_pulse=1*dev.t_pulse)
    # dev.set_vals(Ms_295=0.1*dev.Ms_295, t_pulse=5*dev.t_pulse)
  
  # Perform scurve check
  start_time = time.time()
  validity, scurve = dev_check(dev, plot=True)
  print("Valid:", validity)
  print(f"Runtime: {(time.time() - start_time)} seconds")