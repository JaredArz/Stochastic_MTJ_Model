import sys
# import functions to test directly
sys.path.append('../')
from mtj_types_v3 import MTJ, SHE_MTJ_rng, SWrite_MTJ_rng

# declaration
dev1 = SHE_MTJ_rng()
dev1.set_vals(0)
print(dev1)
dev1.set_mag_vector(3,3)
dev1.set_vals(Ki=8000, tf=1000)
dev1.set_vals(1)
print(dev1)
dev1.set_vals(1)
print(dev1)
dev1.set_vals(1)
print(dev1)
dev1.set_vals(1)
print(dev1)




