# ===== handles fortran interface =====
from interface_funcs import mtj_sample
# ===========================================================
from mtj_types_v3 import SHE_MTJ_rng, VCMA_MTJ_rng, SWrite_MTJ_rng
import sys
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

#plt.style.use(['science','ieee','no-latex'])
colors = [
          '#FFC20A',
          '#0C7BDC',
         ]
cycles = 20
steps = 1
reps  = 2

if len(sys.argv) != 2:
    print("Call with mtj type arg")
    raise(IndexError)
mtj_type = sys.argv[1]

if mtj_type == 'she':
    #apply Hy field, assisted switcihng
    dev = SHE_MTJ_rng()
    J_stt = np.linspace(0,0,steps)
elif mtj_type == 'vcma':
    #try increasing voltage pulse to get full switcihng
    dev = VCMA_MTJ_rng()
    J_stt = np.linspace(0,0,steps)
elif mtj_type == 'swrite':
    dev = SWrite_MTJ_rng()
    Happl = np.linspace(0,150e4,steps)
    J_stt = np.linspace(-300e9,0,steps)
else:
    print("no mtj type of that kind")
    raise(NotImplementedError)

dev.set_vals(0)
dev.set_vals(t_pulse=50e-9,t_relax=50e-9)

def main():
    theta_cyc = []
    phi_cyc = []
    plot_init('phi',1)
    plot_init('theta',2)
    print(dev)
    print("cycling")
    # J_she, v_pulse as set in device parameters
    for rep in range(reps):
        for id,j in enumerate(J_stt):
            dev.set_mag_vector(0, np.pi/2)
            for i in range(cycles):
                # Only tracking magnetization vector stored in *History
                _,_ = mtj_sample(dev,J_stt,1,1)
                for t in dev.thetaHistory:
                    theta_cyc.append(t % 2*3.14159)
                for p in dev.phiHistory:
                    phi_cyc.append(p % 2*3.14159)
        plot(phi_cyc,1,colors[rep])
        plot(theta_cyc,2,colors[rep])
        phi_cyc = []
        theta_cyc = []
    save("phi",1)
    save("theta",2)

def plot_init(name,fig):
    plt.figure(fig)
    plt.tight_layout()
    plt.xlabel('ns')
    plt.ylabel('radians')
    plt.title(f'{mtj_type} {name}')

def plot(arr,fig,color):
    plt.figure(fig)
    plt.plot(arr, color=color)

def save(name,fig):
    plt.figure(fig)
    plt.savefig(f"./results/{mtj_type}_{name}_cycs.png", dpi=1200, format='png')
    plt.close()

if __name__ == "__main__":
    main()
