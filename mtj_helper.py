import numpy as np
from interface_funcs import mtj_sample
import math

def draw_norm(x,psig):
    return (x*np.random.normal(1,psig))

def draw_const(x,csig):
    return (x+np.random.normal(-csig,csig))

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
    updated_val = draw_norm(current_val, stddev)
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

def gamma_pdf(g1, g2, nrange) -> list:
  # Build an analytical gamma probability density function (PDF)

  # g1 corresponds to alpha in gamma distribution definitinon
  # g2 corresponds to beta in gamma distribution, or lmda in previous work here
  # g1 must be an integer for this formula to work. if non-integer g1 are desired, factorial function should become gamma function
  xxis = []
  pdf = []
  for j in range(nrange):
    gval = pow(j,g1-1)*pow(g2,g1)*np.exp(-g2*j)/factorial(g1-1)
    xxis.append(j)
    pdf.append(gval)

  # Normalize exponential distribution
  pdfsum = 0
  for j in range(nrange):
    pdfsum += pdf[j]

  pdf = pdf/pdfsum

  return pdf

def avg_weight_across_samples(dev, apply, samples_to_avg) -> float:
    sum_p = np.sum( [ (mtj_sample(dev, apply),) for _ in range(samples_to_avg)] )
    return sum_p/samples_to_avg
