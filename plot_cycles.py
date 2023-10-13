# ===== handles fortran interface =====
from interface_funcs import mtj_sample
# ===========================================================
from mtj_types_v3 import SHE_MTJ_rng, VCMA_MTJ_rng, SWrite_MTJ_rng, draw_norm
import sys
import matplotlib.ticker as ticker
import math as m
import matplotlib.pyplot as plt
import numpy as np

room_temp = 300
vary_temp_bool = True

colors = [
          '#FFC20A',
          '#0C7BDC',
         ]
cycles = 20
reps   = 2
t_step = 5e-11

if len(sys.argv) != 2:
    print("Call with mtj type arg")
    raise(IndexError)
mtj_type = sys.argv[1]

if mtj_type == 'she':
    dev = SHE_MTJ_rng()
    dev.set_vals(0)
    J_stt = 0
elif mtj_type == 'vcma':
    dev = VCMA_MTJ_rng()
    dev.set_vals(0)
    J_stt = 0
elif mtj_type == 'swrite':
    dev = SWrite_MTJ_rng()
    dev.set_vals(0)
    J_stt = -1.31818e11
else:
    print("no mtj type of that kind")
    raise(NotImplementedError)


def main():
    mz_cyc = []
    plot_init(1)
    print(dev)
    print("cycling")
    # J_she, v_pulse as set in device parameters
    for rep in range(reps):
        dev.set_mag_vector(0, np.pi/2)
        for _ in range(cycles):
            T = draw_norm(room_temp,vary_temp_bool,0.01)
            # Only tracking magnetization vector stored in *History
            _,_ = mtj_sample(dev,J_stt,view_mag_flag=1,dump_mod=1,T=T)
            mz = np.cos(dev.thetaHistory)
            for val in mz:
                mz_cyc.append(val)
        l = len(mz_cyc)
        plot(mz_cyc,1,colors[rep])
        mz_cyc = []
    save(1)

def plot_init(fig):
    plt.figure(fig)
    plt.tight_layout()
    plt.xlabel('ns')
    plt.ylabel('mz')
    plt.title(f'{mtj_type}')

def plot(arr,fig,color):
    plt.figure(fig)
    scale = (t_step/1e-9)
    x = np.arange(0,len(arr))
    x = x * scale
    plt.plot(x,arr,color=color)

def save(fig):
    plt.figure(fig)
    plt.savefig(f"{mtj_type}_mz_cycs.png", dpi=1200, format='png')
    plt.close()

if __name__ == "__main__":
    main()
