#%%
from interface_funcs import mtj_sample
#from mtj_configtest import mtj_mod
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_m_vec_comps(dev):
    mx = np.sin(dev.thetaHistory)*np.cos(dev.phiHistory)
    my = np.sin(dev.thetaHistory)*np.sin(dev.phiHistory)
    mz = np.cos(dev.thetaHistory)
    return mx,my,mz

def configuration_check(dev):
    steps   = 1
    t_step  = 5e-11
    v_pulse = 0
    vhold   = 0
    t_pulse = 10e-9
    t_relax = 15e-9
    Happl = np.linspace(0,0,steps)
    Hshe  = 0
    J_stt = np.linspace(0,0,steps)
    J_she = 5e11
    cycles = 100
    reps   = 1
    pulse_check_start = (1/5) * t_pulse  #10 ns into pulse
    relax_check_start = (3/5) * t_relax #30 ns into relax
    pulse_steps = int(np.floor(t_pulse/t_step))
    relax_steps = int(np.floor(t_relax/t_step))
    pulse_start = int(np.floor(pulse_check_start/t_step))
    relax_start = int(np.floor(relax_check_start/t_step))
    mz_avg = []
    mz_chk1_arr = np.zeros(pulse_steps)
    mz_chk2_arr = np.zeros(relax_steps)
    point2point_variation = 0
    for rep in range(reps):
        for id,j in enumerate(J_stt):
            print('start')
            theta = np.pi/2
            phi = 0
            dev.set_mag_vector(phi,theta)
            mz_arr = []
            for i in tqdm(range(cycles),ncols=80,leave=False):
                _,_ = mtj_sample(dev,J_stt,1,1,i,1)
                mx,my,mz = get_m_vec_comps(dev)

                mz_arr.append(mz)
                mz_chk1_arr = mz_chk1_arr + np.absolute(mz[0:pulse_steps])
                mz_chk2_arr = mz_chk2_arr + np.absolute(mz[pulse_steps:pulse_steps+relax_steps])
                point2point_variation = point2point_variation + np.sum(np.absolute(mz[1::] - mz[0:-1]))/(mz.size - 1)
            mz_avg.append(np.mean(mz_arr))
    mz_chk1_arr = mz_chk1_arr/cycles
    mz_chk2_arr = mz_chk2_arr/cycles
    point2point_variation = point2point_variation/cycles
    mz_chk1_val = np.average(mz_chk1_arr[pulse_start:pulse_steps])
    mz_chk2_val = np.average(mz_chk2_arr[relax_start:relax_steps])
    
    if point2point_variation > 0.25:
        numerical_err = -1
    else:
        numerical_err = 0
    
    if mz_chk1_val < 0.2:
        mz_chk1_res = 0
    elif mz_chk1_val < 0.5:
        mz_chk1_res = 1
    else:
        mz_chk1_res = -1  
        
    if mz_chk2_val < 0.2:
        mz_chk2_res = -1
    elif mz_chk2_val < 0.5:
        mz_chk2_res = 1
    else:
        mz_chk2_res = 0  
        
    if mz_chk1_res == -1:
        PMAIMA = -1 #if check 1 fails, then it is too PMA
    elif mz_chk2_res == -1:
        PMAIMA = 1 #if check 2 fails, it is not PMA enough
    else:
        PMAIMA = 0 #if both checks are just warnings or better, balance is good enough

    plt.plot( np.linspace(0,1,relax_steps+pulse_steps) , mz_arr[0])
    plt.show()

    return numerical_err, mz_chk1_res,mz_chk2_res,PMAIMA
