import sys
# import functions to test directly
sys.path.append('../')
from mtj_types_v3 import MTJ, SHE_MTJ_rng, SWrite_MTJ_rng

# declaration
dev1 = SWrite_MTJ_rng()
dev2 = SHE_MTJ_rng()
dev1.set_vals(Ki = 10)
print(dev1)

dev2.set_vals(1)
print(dev2)

print(dev1)
