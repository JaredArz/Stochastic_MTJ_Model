from datetime import datetime
import os
import multiprocessing as mp

def main():
    hist = []
    bitstream = []
    energy_avg = []
    samples=10
    multirun(function,samples,hist,bitstream,energy_avg)
    print(hist)
    print(bitstream)
    print(energy_avg)

def function(hq,bq,eq):
    now = datetime.now()
    time = now.strftime("%M:%S:%f")
    H = 'H_' + time
    B = 'B_' + time
    E = 'E_' + time
    hq.put(H)
    bq.put(B)
    eq.put(E)

def multirun(f,samples,hist,bitstream,energy_avg):
    batch_size = os.cpu_count() 
    samples_to_run = samples
    while samples_to_run >= 1:        
      if samples_to_run < batch_size:
          batch_size = samples_to_run
      hist_queue       = mp.Queue()  # parallel-safe queue
      bitstream_queue  = mp.Queue()  # parallel-safe queue
      energy_avg_queue = mp.Queue()  # parallel-safe queue
      processes = []
      #   create processes and start them
      for _ in range(batch_size):
          #FIXME
          sim = mp.Process(target=f, args=(hist_queue,bitstream_queue,energy_avg_queue))
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

if __name__ == "__main__":
    main()





