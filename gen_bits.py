# ===== handles fortran interface =====
from interface_funcs import mtj_sample
# ===========================================================
from mtj_types_v3 import SWrite_MTJ_rng, draw_norm
import sys
import math as m
import numpy as np
import time



num_bits = 1e5
J_stt = -1.31818e11
room_temp = 300
vary_temp_bool = False

def main():
    dev  = SWrite_MTJ_rng()
    # create single device with a slight variation on default parameter values
    dev.set_vals(0)
    dev.set_mag_vector()
    print(dev)
    print("Generating bits")
    start_time = time.time()
    bits = []
    for _ in range(int(num_bits)):
        T = draw_norm(room_temp,vary_temp_bool,0.01)
        # need to set upon each sample
        #dev.set_mag_vector()
        bit,energy = mtj_sample(dev,J_stt,T=T)
        bits.append(bit)
    end_time = time.time()

    np.savetxt('./bits.txt', bits, fmt='%i', delimiter=' ', newline='\n',)

    print(f"--- %s seconds per {num_bits} bits --- {end_time-start_time}")

if __name__ == "__main__":
    main()
