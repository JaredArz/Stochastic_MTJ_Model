import sys
from mtj_types_v3 import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
from interface_funcs import mtj_sample, mtj_check, print_check

dev = SHE_MTJ_rng()
dev.set_vals()
dev.set_mag_vector()
print(dev)
print_check(*mtj_check(dev, 20, 250))


