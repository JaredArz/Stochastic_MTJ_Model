import numpy as np

num_devices=20
num_steps = 41
bit_avgs_devices = np.empty((num_devices,num_steps))
nrg_avgs_devices = np.empty((num_devices,num_steps))

for i in range(num_devices):
    dev_i_bits = np.load(f"ooshe_bitavg_{i}.npy")
    dev_i_bits = dev_i_bits.flatten()
    bit_avgs_devices[i,:] = dev_i_bits
for i in range(num_devices):
    dev_i_nrg = np.load(f"ooshe_nrgavg_{i}.npy")
    dev_i_nrg = dev_i_nrg.flatten()
    nrg_avgs_devices[i,:] = dev_i_nrg

print(np.average(bit_avgs_devices,axis=0))
print(np.average(nrg_avgs_devices,axis=0))
