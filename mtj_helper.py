import numpy as np
from interface_funcs import mtj_sample
import math

def draw_gauss(x,psig):
    return (x*np.random.normal(1,psig))


# Device-to-device variation and cycle-to-cycle variation
# can be modeled simply with this function.
# =======
# Takes a device, device parameter, and a percent deviation
# (which defines the standard deviation of a sampled
# gaussian distribution around the current parameter value).
# =======
# Returns the modified device.
def vary_param(dev, param, stddev):
    current_val = dev.__getattribute__(param)
    updated_val = draw_gauss(current_val, stddev)
    dev.__setattr__(param,updated_val)
    return dev


def print_check(nerr, mz1, mz2, PI):
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
  return


def valid_config(nerr, mz1, mz2, PI):
  if nerr == -1:  # Numerical error, do not use parameters
    return False
  elif PI == -1:  # PMA too strong
    return False
  elif PI == 1:   # IMA too strong
    return False
  else:           # Parameters valid
    return True


def gamma_pdf(g1, g2, nrange) -> list:
  # Build an analytical gamma probability density function (PDF)

  # g1 corresponds to alpha in gamma distribution definitinon
  # g2 corresponds to beta in gamma distribution, or lmda in previous work here
  # g1 must be an integer for this formula to work. if non-integer g1 are desired, factorial function should become gamma function
  xxis = []
  pdf = []
  for j in range(nrange):
    gval = pow(j,g1-1)*pow(g2,g1)*np.exp(-g2*j)/math.factorial(g1-1)
    xxis.append(j)
    pdf.append(gval)

  # Normalize exponential distribution
  pdfsum = 0
  for j in range(nrange):
    pdfsum += pdf[j]

  pdf = pdf/pdfsum

  return xxis, pdf


def avg_weight_across_samples(dev, apply, samples_to_avg) -> float:
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


cdf = lambda x,lmda: 1-np.exp(-lmda*x)
def dist_rng(dev,k,init,lmda,dump_mod_val,mag_view_flag,file_ID,jz_lut_func):
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
    # out,energy = mtj_sample(dev,jz_lut_she(pright),dump_mod_val,mag_view_flag,file_ID)
    out,energy = mtj_sample(dev,jz_lut_func(pright),mag_view_flag,dump_mod_val,file_ID)
    bits.append(out)
    energies.append(energy)

    if out == 1:
      x0 = x1
    elif out == 0:
      x2 = x1
    x1 = (x2+x0)/2
    number += out*2**(k-i-1)
  return number,bits,energies


def get_energy(dev, samples, jz_lut_func):
  k       = 8
  lmda    = 0.01
  init_t  = 9*np.pi/10
  number_history = []
  bitstream  = []
  energy_avg = []
  mag_view_flag = False
  dump_mod_val  = 8000

  for j in range(samples):
    number_j,bits_j,energies_j = dist_rng(dev,k,init_t,lmda,dump_mod_val,mag_view_flag,j+7,jz_lut_func)
    number_history.append(number_j)
    bitstream.append(''.join(str(i) for i in bits_j))
    energy_avg.append(np.average(energies_j))
  
  return number_history, bitstream, energy_avg