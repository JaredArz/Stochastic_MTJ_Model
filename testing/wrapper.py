import sys
sys.path.append('./fortran_source')
import single_sample as ss
import mtj_types as mtj
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def main():
    th_init =  np.pi/2
    ph_init = np.pi/2
    num_iter = 1000
    J_she = 1e-3
    Jappl = 1e-3
    device_f = mtj.SHE_MTJ_rng(th_init,ph_init,0)
    ki_f  = device_f.Ki
    tmr_f = device_f.TMR
    rp_f  = device_f.Rp

    view_theta_and_phi_flag = True
    for i in range(10):
        e_f,_,theta_f,phi_f = ss.single_sample(Jappl,J_she,device_f.theta,device_f.phi,ki_f,tmr_f,rp_f,view_theta_and_phi_flag)
        if(view_theta_and_phi_flag):
            view_theta_and_phi_flag = False
    array  = np.loadtxt("time_evol_phi.txt", dtype=float, delimiter=None, skiprows=0, max_rows=1)
    array2 = np.loadtxt("time_evol_theta.txt", dtype=float, delimiter=None, skiprows=0, max_rows=1)


    date   = datetime.now().strftime("%m-%d_%H:%M:%S")
    w_dir,plot_w_path = handle_w_paths(date)
    move_text_files_to_permanent(date,w_dir)

    num_steps = len(array)
    x_axis    = np.linspace(0,(5e-11)*(num_steps),num_steps)
    figure, axis = plt.subplots(nrows=1, ncols=2)
    axis[0].plot(x_axis,array)
    axis[1].plot(x_axis,array2)
    axis[0].set_title("phi")
    axis[1].set_title("theta")
    figure.tight_layout()
    plt.savefig(plot_w_path,format='png',dpi=1200)

def sample_neuron(devs,Jstt,Jsot,view_mag_flag):
        energy, bit, theta_end, phi_end = ss.single_sample.pulse_then_relax(Jstt,Jsot,\
                                                      devs[h].theta,devs[h].phi,       \
                                                      devs[h].Ki,devs[h].TMR,devs[h].Rp,view_mag_flag)
        devs[h].theta = theta_end
        devs[h].phi = phi_end
        return bit

def move_text_files_to_permanent(date, w_dir):
    new_theta_path = Path(str(w_dir) + '/' + 'time_evol_theta_' + date +'.txt')
    new_phi_path = Path(str(w_dir) + '/' + 'time_evol_phi_' + date +'.txt')
    os.rename('time_evol_phi.txt',  str(new_phi_path))
    os.rename( 'time_evol_theta.txt', str(new_theta_path))

def handle_w_paths(date):
    #pathlib POSIX path creation
    out_dir_path = Path("./outputs")
    if not os.path.isdir(out_dir_path):
        os.mkdir(out_dir_path)
    w_dir_path = Path("./outputs/time_evol_mag_"+date)
    if not os.path.isdir(w_dir_path):
        os.mkdir(w_dir_path)
    plot_w_file = ("time_evol_mag_" + date + '.png') 
    plot_w_path = Path(w_dir_path / plot_w_file)
    return(w_dir_path,plot_w_path)



if __name__ == "__main__":
    main()
