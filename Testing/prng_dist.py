import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
from scipy import stats
from scipy.special import rel_entr


SAMPLES = 100_000
BITS = 8
GAMMA_LOWER = 0.1
GAMMA_UPPER = 0.24
RESOLUTION = 256*2

CDF_XXIS = np.linspace(GAMMA_LOWER, GAMMA_UPPER, RESOLUTION)
CDF_ARR = stats.gamma.cdf(CDF_XXIS, a=50, scale=1/311.44)


def remap(num, inMin, inMax, outMin, outMax):
  rv = outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))
  return int(rv)


def get_pdf(type="exp"):
  if type == "exp":
    xxis = np.linspace(0, 255, 256)
    pdf = stats.gamma.pdf(xxis, a=1, scale=1/0.01)
    pdf = pdf/np.sum(pdf)
  elif type == "gamma":
    xxis = np.linspace(GAMMA_LOWER, GAMMA_UPPER, 256)
    pdf = stats.gamma.pdf(xxis, a=50, scale=1/311.44)
    pdf = pdf/np.sum(pdf)
  else:
    raise TypeError("Invalid pdf type")
  
  return xxis, pdf


def get_bits():
  cdf = lambda x: CDF_ARR[round(remap(x, GAMMA_LOWER, GAMMA_UPPER, 0, len(CDF_ARR)-1))]
  
  x2 = GAMMA_UPPER
  x0 = GAMMA_LOWER
  x1 = (x2+x0)/2
  number = 0
  bits = []

  for i in range(BITS):
    pright = (cdf(x2)-cdf(x1))/(cdf(x2)-cdf(x0))
    
    if random.random() < pright:
      out = 1
      x0 = x1
    else:
      out = 0
      x2 = x1
  
    x1 = (x2+x0)/2
    number += out*2**(BITS-i-1)
    bits.append(out)

  return number, bits


def prng_dist(samples=SAMPLES):
  num_history = []
  bit_history = []

  for i in range(samples):
    num, bit = get_bits()
    num_history.append(num)
    bit_history.append(bit)
  
  # Build gamma distribution
  xxis, pdf = get_pdf("gamma")

  # Calculate chi2
  counts, _ = np.histogram(num_history, bins=256)
  pdf = pdf*samples
  chi2 = 0
  for j in range(256):
    chi2 += ((counts[j]-pdf[j])**2)/pdf[j]
  counts = counts/samples
  pdf = pdf/samples

  # Calculate KL-Div
  kl_div_score = sum(rel_entr(counts, pdf))

  # print("Chi2  :", chi2)
  # print("KL_Div:", kl_div_score)

  return xxis, pdf, counts