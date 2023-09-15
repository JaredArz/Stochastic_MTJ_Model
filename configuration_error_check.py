#%%
from mtj_configtest import mtj_mod
import numpy as np
from tqdm import tqdm

def configuration_check(TMR, Ki, Rp, Ms):
    steps = 1
    t_step = 5e-11
    v_pulse = 0
    vhold = 0
    t_pulse = 50e-9
    t_relax = 50e-9
    Happl = np.linspace(0,0,steps)
    Hshe = 0 # 300Oe=2.4e4 200Oe=1.6e4 100Oe=8e3
    J_stt = np.linspace(0,0,steps)
    J_she = -4e11
    cycles = 100
    reps = 1
    pulse_check_start = 10e-9 #10 ns into pulse
    relax_check_start = 30e-9 #30 ns into relax
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
            print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}')
            theta = np.pi/2
            phi = 0
            mz_arr = []
            for cy in tqdm(range(cycles),ncols=80,leave=False):
                theta,phi,t,r,g,mx,my,mz,bitstr,_,energy = mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold, TMR, Ki, Rp, Ms)
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

    return numerical_err, mz_chk1_res,mz_chk2_res,PMAIMA
