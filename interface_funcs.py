import sys
sys.path.append("./fortran_source")
import single_sample as f90
import os
import signal
import numpy as np

def mtj_sample(dev,Jstt,dump_mod,view_mag_flag,file_ID) -> (int,float):
        if dev.theta is None or dev.phi is None or dev.params_set_flag is None:
            print("\nMag vector or device parameters not initialized, exititng.")
            os.kill(os.getppid(), signal.SIGTERM)
        else:
            # fortran call here.
            energy, bit, theta_end, phi_end = f90.single_sample.pulse_then_relax(Jstt,\
                    dev.J_she,dev.theta,dev.phi,dev.Ki,dev.TMR,dev.Rp,\
                    dev.a,dev.b,dev.tf,dev.alpha,dev.Ms,dev.eta,dev.d,\
                    dev.t_pulse,dev.t_relax,\
                    dump_mod,view_mag_flag,dev.sample_count,file_ID)
            # Need to update device objects and put together time evolution data after return.
            dev.set_mag_vector(phi_end,theta_end)
            if(view_mag_flag and (dev.sample_count % dump_mod == 0)):
                # These file names are determined by fortran subroutine single_sample.
                theta_from_txt = np.loadtxt("time_evol_mag_"+ format_file_ID(file_ID) + ".txt", dtype=float, delimiter=None, skiprows=0, max_rows=1)
                phi_from_txt   = np.loadtxt("time_evol_mag_"+ format_file_ID(file_ID) + ".txt", dtype=float, delimiter=None, skiprows=1, max_rows=1)
                os.remove("time_evol_mag_" + format_file_ID(file_ID) + ".txt")
                dev.thetaHistory.append(list(theta_from_txt))
                dev.phiHistory.append(list(phi_from_txt))
            if(view_mag_flag):
                dev.sample_count+=1
            return bit,energy

# FIXME: currently not working.
"""
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

# do not change, format to length four with 0's to the left
def format_file_ID(pid) -> str:
    str_pid = str(pid)
    while len(str_pid) < 7:
        str_pid = '0' + str_pid
    return str_pid
