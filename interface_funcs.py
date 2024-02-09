import sys
import re
# extract absolute path of main dir from entry dir to locate 'fortran_source' properly
main_dir = re.match(re.compile(r'^(.*?Stochastic_MTJ_Model)'), sys.path[0]).group(1)
sys.path.append(f"{main_dir}/fortran_source")
import sampling as f90
import os
import numpy as np

# NOTE: assumes ohmic relationship
def V_to_J(dev, V):
    return V/dev.RA

def mtj_check(dev, V, cycles, pcs = None, rcs = None) -> (int,float):
    if dev.heating_capable == 0 and dev.heating_enabled == 1:
        raise(AttributeError)
    try:
        if pcs == None or rcs == None:
            pcs = (1/5) * dev.t_pulse
            rcs = (3/5) * dev.t_relax
        mz_c1, mz_c2, p2pv = f90.sampling.check_she(V_to_J(dev,V),\
                dev.J_she, dev.Hy, dev.theta, dev.phi, dev.Ki, dev.TMR, dev.Rp,\
                dev.a, dev.b, dev.tf, dev.alpha, dev.Ms, dev.eta, dev.d, dev.tox,\
                dev.t_pulse, dev.t_relax, dev.T,\
                dev.heating_enabled, cycles, pcs, rcs)
    except(AttributeError):
        dev.print_init_error()
        raise

    mz_chk1_res = None
    mz_chk2_res = None

    if p2pv > 0.25:
        nerror = -1
    else:
        nerror = 0
        if mz_c1 < 0.2:
            mz_chk1_res = 0
        elif mz_c1 < 0.5:
            mz_chk1_res = 1
        else: mz_chk1_res = -1

        if mz_c2 < 0.2:
            mz_chk2_res = -1
        elif mz_c2 < 0.5:
            mz_chk2_res = 1
        else: mz_chk2_res = 0

        PI = 0
        if mz_chk1_res == -1:
            PI = -1
        elif mz_chk2_res == -1:
            PI = 1
    return nerr, mz_chk1_res, mz_chk2_res, PI

def mtj_sample(dev, V, view_mag_flag = 0, dump_mod = 1,
               file_ID = 1) -> (int,float):
    if dev.heating_capable == 0 and dev.heating_enabled == 1:
        raise(AttributeError)
    try:
        # fortran call here.
        if (dev.mtj_type == 0):
            #energy, bit, theta_end, phi_end = f90.sampling.sample_she(V_to_J(dev,V),\
            energy, bit, theta_end, phi_end = f90.sampling.sample_she(V_to_J(dev,V),\
                    dev.J_she, dev.Hy, dev.theta, dev.phi, dev.Ki, dev.TMR, dev.Rp,\
                    dev.a, dev.b, dev.tf, dev.alpha, dev.Ms, dev.eta, dev.d, dev.tox,\
                    dev.t_pulse, dev.t_relax, dev.T,\
                    dump_mod, view_mag_flag, dev.sample_count, file_ID,\
                    dev.heating_enabled)
        elif (dev.mtj_type == 1):
            energy, bit, theta_end, phi_end = f90.sampling.sample_swrite(V_to_J(dev,V),\
                    dev.J_reset, dev.H_reset, dev.theta, dev.phi, dev.K_295, dev.TMR, dev.Rp,\
                    dev.a, dev.b, dev.tf, dev.alpha, dev.Ms_295, dev.eta, dev.d, dev.tox,\
                    dev.t_pulse, dev.t_relax, dev.t_reset, dev.T,\
                    dump_mod, view_mag_flag, dev.sample_count, file_ID,\
                    dev.heating_enabled)

        elif (dev.mtj_type == 2):
            energy, bit, theta_end, phi_end = f90.sampling.sample_vcma(V_to_J(dev,V),\
                    dev.v_pulse, dev.theta, dev.phi, dev.Ki, dev.TMR, dev.Rp,\
                    dev.a, dev.b, dev.tf, dev.alpha, dev.Ms, dev.eta, dev.d, dev.tox,\
                    dev.t_pulse, dev.t_relax, dev.T,\
                    dump_mod, view_mag_flag, dev.sample_count, file_ID,\
                    dev.heating_enabled)
        else:
            dev.print_init_error()
            raise(AttributeError)
        # Need to update device objects and put together time evolution data after return.
        dev.set_mag_vector(phi_end,theta_end)
        if( (view_mag_flag and (dev.sample_count % dump_mod == 0))):
            # These file names are determined by fortran subroutine single_sample.
            phi_from_txt   = np.loadtxt("phi_time_evol_"+ format_file_ID(file_ID) + ".txt",
                                        dtype=float, usecols=0, delimiter=None)
            theta_from_txt = np.loadtxt("theta_time_evol_"+ format_file_ID(file_ID) + ".txt",
                                        dtype=float, usecols=0, delimiter=None)
            temp_from_txt  = np.loadtxt("temp_time_evol_"+ format_file_ID(file_ID) + ".txt",
                                        dtype=float, usecols=0, delimiter=None)
            os.remove("phi_time_evol_"   + format_file_ID(file_ID) + ".txt")
            os.remove("theta_time_evol_" + format_file_ID(file_ID) + ".txt")
            os.remove("temp_time_evol_" + format_file_ID(file_ID) + ".txt")
            dev.thetaHistory = list(theta_from_txt)
            dev.phiHistory   = list(phi_from_txt)
            dev.tempHistory  = list(temp_from_txt)
        if(view_mag_flag):
            dev.sample_count+=1
        return bit,energy
    except(AttributeError):
        dev.print_init_error()
        raise

# Format must be consistent with fortrn, do not change
# File ID of length seven with 0's to the left
def format_file_ID(pid) -> str:
    str_pid = str(pid)
    while len(str_pid) < 7:
        str_pid = '0' + str_pid
    return str_pid
