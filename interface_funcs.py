import sys
sys.path.append("./fortran_source")
import single_sample as f90
import os
import signal
import numpy as np
import multiprocessing as mp

def run_in_parallel_batch(func,samples,\
                            dev,k,init,lmda,hist,bitstream,energy_avg,\
                            mag_view_flag,batch_size=None) -> (list,list,list):
  if batch_size is None:
      #f = open("cpu_count.txt",'a')
      #f.write(str(os.cpu_count()) + "\n")
      #f.close
      batch_size = os.cpu_count() 
      #batch_size = 128
  else:
      #FIXME:
      pass
  samples_to_run = samples
  while samples_to_run >= 1:        
      if samples_to_run < batch_size:
          batch_size = samples_to_run
      hist_queue       = mp.Queue()  # parallel-safe queue
      bitstream_queue  = mp.Queue()  # parallel-safe queue
      energy_avg_queue = mp.Queue()  # parallel-safe queue
      processes = []
      #   create processes and start them
      proc_IDs = generate_proc_IDs(batch_size)
      for _ in range(batch_size):
          sim = mp.Process(target=func, args=(dev,k,init,lmda,hist_queue,bitstream_queue,energy_avg_queue,\
                                                        mag_view_flag,proc_IDs[_]))
          processes.append(sim)
          sim.start()
      #   waits for solution to be available
      for sim in processes:
          single_hist      = hist_queue.get() 
          single_bitstream = bitstream_queue.get()  
          single_energy    = energy_avg_queue.get() 
          hist.append(single_hist)
          bitstream.append(single_bitstream)
          energy_avg.append(single_energy)
      #   wait for all processes to wrap-up before continuing
      for sim in processes:
          sim.join()
      samples_to_run -= batch_size
  return hist,bitstream,energy_avg

def mtj_sample(dev,Jstt,view_mag_flag,proc_ID) -> (int,float):
        if dev.theta is None or dev.phi is None or dev.params_set_flag is None:
            print("\nMag vector or device parameters not initialized, exititng.")
            os.kill(os.getppid(), signal.SIGTERM)
        else:
            # fortran call here.
            energy, bit, theta_end, phi_end = f90.single_sample.pulse_then_relax(Jstt,\
                    dev.J_she,dev.theta,dev.phi,dev.Ki,dev.TMR,dev.Rp,\
                    dev.a,dev.b,dev.tf,dev.alpha,dev.Ms,dev.eta,dev.d,\
                    view_mag_flag,proc_ID)
            # Need to update device objects and put together time evolution data after return.
            dev.set_mag_vector(phi_end,theta_end)
            if(view_mag_flag):
                # These file names are determined by fortran subroutine single_sample.
                theta_from_txt = np.loadtxt("time_evol_mag_"+ format_proc_ID(proc_ID) + ".txt", dtype=float, delimiter=None, skiprows=0, max_rows=1)
                phi_from_txt   = np.loadtxt("time_evol_mag_"+ format_proc_ID(proc_ID) + ".txt", dtype=float, delimiter=None, skiprows=1, max_rows=1)
                os.remove("time_evol_mag_" + format_proc_ID(proc_ID) + ".txt")
                dev.thetaHistory.append(list(theta_from_txt))
                dev.phiHistory.append(list(phi_from_txt))
            return bit,energy

# generate random number string and check to ensure all are unique
def generate_proc_IDs(batch_size) -> list:
    # important for prod ID to not be 6 for fortran interoperability
    proc_IDs = (np.random.uniform(7,9999999,size=(1,batch_size)))[0]
    set_of_IDs = set(proc_IDs)
    if(len(set_of_IDs) < batch_size):  
        return generate_proc_IDs(batch_size)
    else:
        return [int(ID) for ID in proc_IDs]

# format to length four with 0's to the left
def format_proc_ID(pid) -> str:
    str_pid = str(pid)
    while len(str_pid) < 7:
        str_pid = '0' + str_pid
    return str_pid
