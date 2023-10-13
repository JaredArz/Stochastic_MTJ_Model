# ===== handles fortran interface =====
from interface_funcs import mtj_sample
# ===========================================================
from mtj_types_v3 import SWrite_MTJ_rng
import sys
import math as m
import numpy as np
import time

cycles = 1000000
reps = 10
t_step = 5e-11
J_stt = -1.31818e11

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

if __name__ == "__main__":
    main()
