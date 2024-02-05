import sys
from mtj_types_v3 import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
from interface_funcs import mtj_sample

dev = VCMA_MTJ_rng()
dev.set_vals(1)
dev.set_mag_vector()
#dev.enable_heating()
print(dev)
_,_ = mtj_sample(dev, -1.1, view_mag_flag = True)
print(dev.tempHistory)
