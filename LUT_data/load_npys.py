import numpy as np
import glob

bitavgs = glob.glob('*bitavg_*.npy')
nrgavgs = glob.glob('*nrgavg_*.npy')
#num_devices=20
num_steps = 41
bit_avgs_devices = np.empty((len(bitavgs),num_steps))
nrg_avgs_devices = np.empty((len(nrgavgs),num_steps))

for f in enumerate(bitavgs):
    dev_i_bits = np.load(f[1])
    dev_i_bits = dev_i_bits.flatten()
    bit_avgs_devices[f[0],:] = dev_i_bits
for f in enumerate(nrgavgs):
    dev_i_nrg = np.load(f[1])
    dev_i_nrg = dev_i_nrg.flatten()
    nrg_avgs_devices[f[0],:] = dev_i_nrg

print(list(np.average(bit_avgs_devices,axis=0)))
print(list(np.average(nrg_avgs_devices,axis=0)))
