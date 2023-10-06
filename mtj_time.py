# ===== handles fortran interface =====
from interface_funcs import mtj_sample
# ===========================================================
from mtj_types_v3 import SWrite_MTJ_rng
import sys
import math as m
import matplotlib.pyplot as plt
import numpy as np
import time

colors = [
          '#FFC20A',
          '#0C7BDC',
         ]
cycles = 1000000
reps = 10
t_step = 5e-11
J_stt = -2.5e11

def main():
    dev  = SWrite_MTJ_rng()
    dev.set_vals(0)
    dev.set_mag_vector()
    print(dev)
    print("cycling")
    times = [0,0,0,0,0,0,0,0,0,0]
    for rep in range(reps):
        start_time = time.time()
        for _ in range(cycles):
            _,_ = mtj_sample(dev,J_stt)
        times[rep] = (time.time() - start_time)

    print(f"--- %s seconds per {cycles} --- {np.average(times)}")

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
