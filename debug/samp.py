import sys
sys.path.append('../')
from mtj_types_v3 import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
from mtj_variation import vary_param
from interface_funcs import mtj_sample


dev = SWrite_MTJ_rng("NYU")
dev.set_vals(1)
print(dev)
dev.set_mag_vector()
bit,e = mtj_sample(dev,5e11)
print(bit)
print(dev.tempHistory)
