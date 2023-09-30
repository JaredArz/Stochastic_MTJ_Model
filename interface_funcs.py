import sys
sys.path.append("./fortran_source")
import sampling as f90
import os
import signal
import numpy as np

def mtj_sample(dev,Jstt,dump_mod=1,view_mag_flag=0,file_ID=1,config_check=0) -> (int,float):
    try:
        # fortran call here.
        if (dev.mtj_type == 0):
            energy, bit, theta_end, phi_end = f90.sampling.sample_she(Jstt,\
                    dev.J_she, dev.theta, dev.phi, dev.Ki, dev.TMR, dev.Rp,\
                    dev.a, dev.b, dev.tf, dev.alpha, dev.Ms, dev.eta, dev.d,\
                    dev.t_pulse, dev.t_relax,\
                    dump_mod, view_mag_flag, dev.sample_count, file_ID, config_check)
        elif (dev.mtj_type == 1):
            energy, bit, theta_end, phi_end = f90.sampling.sample_swrite(Jstt,\
                    dev.J_reset,dev.H_appl,dev.H_reset,dev.theta,dev.phi,dev.Ki,dev.TMR,dev.Rp,\
                    dev.a,dev.b,dev.tf,dev.alpha,dev.Ms,dev.eta,dev.d,\
                    dev.t_pulse,dev.t_relax,dev.t_reset,\
                    dump_mod,view_mag_flag,dev.sample_count,file_ID,config_check)
        elif (dev.mtj_type == 2):
            energy, bit, theta_end, phi_end = f90.sampling.sample_vcma(Jstt,\
                    dev.v_pulse, dev.theta, dev.phi, dev.Ki, dev.TMR, dev.Rp,\
                    dev.a, dev.b, dev.tf, dev.alpha, dev.Ms, dev.eta, dev.d,\
                    dev.t_pulse, dev.t_relax,\
                    dump_mod, view_mag_flag, dev.sample_count, file_ID, config_check)
        else:
            dev.print_init_error()
            raise(AttributeError)
        # Need to update device objects and put together time evolution data after return.
        dev.set_mag_vector(phi_end,theta_end)
        if( (view_mag_flag and (dev.sample_count % dump_mod == 0)) or config_check):
            # These file names are determined by fortran subroutine single_sample.
            phi_from_txt   = np.loadtxt("time_evol_mag_"+ format_file_ID(file_ID) + ".txt", dtype=float, delimiter=None, skiprows=0, max_rows=1)
            theta_from_txt = np.loadtxt("time_evol_mag_"+ format_file_ID(file_ID) + ".txt", dtype=float, delimiter=None, skiprows=1, max_rows=1)
            os.remove("time_evol_mag_" + format_file_ID(file_ID) + ".txt")
            dev.thetaHistory = list(theta_from_txt)
            dev.phiHistory   = list(phi_from_txt)
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

""" ### FIXME: currently not working.
def run_in_parallel_batch(func,samples,\
                            dev,k,init,lmda,\
                            dump_mod,mag_view_flag,batch_size=None) -> (list,list,list):
  if batch_size is None:
      batch_size = 2*os.cpu_count()
  else:
      #FIXME: add better parallelization and error handling
      pass
  args = (dev,k,init,lmda,dump_mod,mag_view_flag)
  func_data = parallel_env(samples,batch_size,func,args).run()
  return func_data[0],func_data[1],func_data[2]
"""
