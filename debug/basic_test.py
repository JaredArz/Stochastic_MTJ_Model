from mtj_types_v3 import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng
from mtj_variation import vary_param



dev = SHE_MTJ_rng()
print(dev)
print(dev.valid_params)
print(dev.dflt_params)
dev.set_vals(1)
dev.set_vals(Ki=1)
dev.set_mag_vector(theta=1)
print(dev)

input()
dev = SWrite_MTJ_rng("UTA")
print(dev)
print(dev.valid_params)
print(dev.dflt_params)
dev.set_vals(1)
dev.set_vals(Ki=1)
dev.set_mag_vector(theta=1)
print(dev)

input()
dev = SWrite_MTJ_rng("NYU")
print(dev)
print(dev.valid_params)
print(dev.dflt_params)
dev.set_vals(1)
dev.set_vals(K_295=1)
dev.set_mag_vector(theta=1)
print(dev)

input()
dev = VCMA_MTJ_rng()
print(dev)
print(dev.valid_params)
print(dev.dflt_params)
dev.set_vals(1)
dev.set_vals(Ki=1)
dev.set_mag_vector(theta=1)
print(dev)

input()

print(dev)
vary_param(dev, 'alpha', 5100)
print(dev)
