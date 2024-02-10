from interface_funcs import mtj_sample
from mtj_types import SWrite_MTJ_rng, SHE_MTJ_rng, VCMA_MTJ_rng
from mtj_helper import avg_weight_across_samples
import numpy as np
import matplotlib.pyplot as plt

def generate_scurve(dev, apply, num_to_avg):
    weights = []
    for j in apply:
      weight = avg_weight_across_samples(dev, j, num_to_avg)
      weights.append(weight)
    return weights

j_steps = 100
num_to_avg = 500
SHE_current    = np.linspace(-6e9, 6e9,j_steps)
SWrite_current = np.linspace(-300e9, 0,j_steps)
VCMA_current   = np.linspace(-6e9, 6e9,j_steps)

SWrite = SWrite_MTJ_rng("UTA")
SWrite.init()
print(SWrite)

SHE = SHE_MTJ_rng()
SHE.init()
print(SHE)

VCMA = VCMA_MTJ_rng()
VCMA.init()
print(VCMA)

SHE_scurve    = generate_scurve(SHE, SHE_current, num_to_avg)
SWrite_scurve = generate_scurve(SWrite, SWrite_current, num_to_avg)
VCMA_scurve   = generate_scurve(VCMA, VCMA_current, num_to_avg)

_, SHE_ax    = plt.subplots()
_, SWrite_ax = plt.subplots()
_, VCMA_ax   = plt.subplots()

SHE_ax.plot(SHE_current,SHE_scurve)
SWrite_ax.plot(SWrite_current,SWrite_scurve)
VCMA_ax.plot(VCMA_scurve,VCMA_scurve)

SHE_ax.set_xlabel('J [A/m^2]')
SWrite_ax.set_xlabel('J [A/m^2]')
VCMA_ax.set_xlabel('J [A/m^2]')

plt.show()
