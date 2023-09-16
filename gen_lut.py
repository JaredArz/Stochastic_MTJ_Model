import sys
sys.path.append("./fortran_source")
from interface_funcs import mtj_sample
from mtj_types_v3 import SHE_MTJ_rng
#============================================================
#from original_mtj_types_Ki_sweep_optimized import SHE_MTJ_rng
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# lookup table generation script
steps = 41
Happl = np.linspace(-2e4,2e4,steps)
Happl = np.zeros_like(Happl)
Jappl = np.linspace(-6e9,6e9,steps)
cycles = 10000
reps = 1
devnum = 20
theta = np.pi/100
phi = 0
for idev in range(devnum):
    dev = SHE_MTJ_rng()
    dev.set_vals(0)
    dev.set_mag_vector(phi,theta)
    bitstr_avg = []
    energy_avg = []
    for rep in range(reps):
        for id,j in enumerate(Jappl):
            print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}, dev {idev}')
            #r_arr = []
            #g_arr = []
            #m_arr = []
            bitstr_arr = []
            energy_arr = []
            for i in tqdm(range(cycles),ncols=80,leave=False):
                bitstr,energy = mtj_sample(dev,j,1,False,i)
                bitstr_arr.append(bitstr)
                energy_arr.append(energy)
            bitstr_avg.append(np.mean(bitstr_arr))
            energy_avg.append(np.sum(energy_arr)/cycles)
            print(f'bitstr_avg = {bitstr_avg[-1]}; energy_avg = {energy_avg[-1]}')
            print('---------------')
    bitstr_avg = np.array(bitstr_avg).reshape(reps,steps).T
    energy_avg = np.array(energy_avg).reshape(reps,steps).T
    np.save(f'ooshe_bitavg_{idev}.npy',bitstr_avg)
    np.save(f'ooshe_nrgavg_{idev}.npy',energy_avg)
    np.save(f'J_appl.npy',Jappl)
    np.save(f'H_appl.npy',Happl)
