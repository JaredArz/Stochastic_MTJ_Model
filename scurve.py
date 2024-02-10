from interface_funcs import mtj_sample
from mtj_types import SWrite_MTJ_rng, SHE_MTJ_rng, VCMA_MTJ_rng
from mtj_helper import avg_weight_across_samples
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

class scurve:
  def __init__(self, dev, x, title):
    num_to_avg = 500
    self.x = x
    self.y = self.generate(dev, num_to_avg)
    fig,ax = plt.subplots()
    ax.set_ylim([-0.1,1.1])
    ax.set_xlabel('J [A/m^2]')
    ax.set_title(title)
    self.ax = ax

  def generate(self, dev, num_to_avg):
      weights = []
      for J in self.x:
        weight = avg_weight_across_samples(dev, J, num_to_avg)
        weights.append(weight)
      return weights

j_steps = 100
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

Ss = [ scurve( SHE, SHE_current, "SHE" ),
       scurve( SWrite, SWrite_current, "SWrite" ),
       scurve( VCMA, VCMA_current, "VCMA" )]

for S in Ss: (S.ax).plot(S.x, S.y)

plt.show()
