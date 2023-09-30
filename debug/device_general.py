import sys
# import functions to test directly
sys.path.append('../')
from mtj_types_v3 import SHE_MTJ_rng

# declaration
dev = SHE_MTJ_rng()

# set (phi,theta)
dev.set_mag_vector(3.14/2, 3.14/2)

# print device parameters
print(dev)
input()

dev.Ki=1
dev.Ms=1
dev.tf=1
dev.J_she=1
dev.a=1
dev.b=1
dev.d=1
dev.eta=1
dev.alpha=1
dev.Rp=1
dev.TMR=1
print(dev)
input()

# set device parameters, a few at a time
dev.set_vals(Ki=1,Ms=1)
print(dev)
input()

# all at once
dev.set_vals(Ki=1,Ms=1,tf=1,J_she=1,a=1,b=1,d=1,eta=1,alpha=1,Rp=1,TMR=1)
print(dev)
input()

# debug option available, uses a known good device
dev.set_vals(True)
print(dev)
input()


