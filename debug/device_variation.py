import sys
# import functions to test directly
sys.path.append('../')
from mtj_types_v3 import SHE_MTJ_rng

# declaration
dev = SHE_MTJ_rng()

# print device parameters
print(dev)
print(dev.params_set_flag)

dev.set_vals(0)
print(dev)

dev.set_vals(1)
print(dev)

