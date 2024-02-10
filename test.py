from interface_funcs import mtj_sample, mtj_check
from mtj_helper import print_check
from mtj_types import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng

dev = SHE_MTJ_rng()
dev.init() # calls both set_vals and set_mag_vector with defaults
print(dev)

print_check(*mtj_check(dev, 0, 100))

dev = SWrite_MTJ_rng("UTA")
dev.init()
print(dev)

print_check(*mtj_check(dev, -2e11, 100))

dev = VCMA_MTJ_rng()
dev.init()
print(dev)

print_check(*mtj_check(dev, 0, 100))
