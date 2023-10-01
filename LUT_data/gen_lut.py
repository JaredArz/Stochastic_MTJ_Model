import sys
sys.path.append("../")
sys.path.append("../fortran_source")
from interface_funcs import mtj_sample
#============================================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Call with mtj type arg")
    raise(IndexError)
mtj_type = sys.argv[1]

steps = 41
cycles = 10000
reps = 1
devnum = 20

if mtj_type == 'she':
    from mtj_types_v3 import SHE_MTJ_rng as MTJ_rng
    Happl = np.linspace(-2e4,2e4,steps)
    Jappl = np.linspace(-6e9,6e9,steps)
    theta = np.pi/100
    phi = 0
elif mtj_type == 'vcma':
    from mtj_types_v3 import VCMA_MTJ_rng as MTJ_rng
    Happl = np.linspace(-2e4,2e4,steps)
    Jappl = np.linspace(-6e9,6e9,steps)
    theta = np.pi/100
    phi = 0
elif mtj_type == 'swrite':
    from mtj_types_v3 import SWrite_MTJ_rng as MTJ_rng
    Happl = np.linspace(0,150e4,steps)
    Jappl = np.linspace(-300e9,0,steps)
    theta = 99*np.pi/100
    phi = 0
else:
    print("no mtj type of that kind")
    raise(NotImplementedError)

Happl = np.zeros_like(Happl)

for idev in range(devnum):
    dev = MTJ_rng()
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
            for _ in tqdm(range(cycles),ncols=80,leave=False):
                bitstr,energy = mtj_sample(dev,j)
                bitstr_arr.append(bitstr)
                energy_arr.append(energy)
            bitstr_avg.append(np.mean(bitstr_arr))
            energy_avg.append(np.sum(energy_arr)/cycles)
            print(f'bitstr_avg = {bitstr_avg[-1]}; energy_avg = {energy_avg[-1]}')
            print('---------------')
    bitstr_avg = np.array(bitstr_avg).reshape(reps,steps).T
    energy_avg = np.array(energy_avg).reshape(reps,steps).T
    np.save(f'oo{mtj_type}_bitavg_{idev}.npy',bitstr_avg)
    np.save(f'oo{mtj_type}_nrgavg_{idev}.npy',energy_avg)
    np.save(f'J_appl.npy',Jappl)
    np.save(f'H_appl.npy',Happl)
