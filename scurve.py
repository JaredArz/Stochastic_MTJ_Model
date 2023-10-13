# ===== handles fortran interface and batch parallelism =====
from interface_funcs import mtj_sample
# ===========================================================
import os
import sys
import time
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from mtj_types_v3 import SWrite_MTJ_rng, SHE_MTJ_rng, draw_norm
import re

def gen():
  start_time = time.time()
  out_path = get_out_path()
  devices = []
  dev_variations = 10
  num_to_avg = 100
  room_temp = 300
  j_steps = 150
  J_lut = np.linspace(-300e9,0,j_steps)

  for _ in range(dev_variations):
    dev = SWrite_MTJ_rng()
    dev.set_vals(1)
    devices.append(dev)

  for dev_num,dev in enumerate(devices):
    weights = []
    for j in range(j_steps):
      avg_wght = 0
      for _ in range(num_to_avg):
        T = draw_norm(room_temp,1,0.01)
        dev.set_mag_vector()
        out,energy = mtj_sample(dev,J_lut[j],T=T)
        avg_wght = avg_wght + out
      avg_wght = avg_wght/num_to_avg
      weights.append(avg_wght)
    w_file = f"{out_path}/weight_data_{dev_num}.txt"
    f = open(w_file,'w')
    for i in range(j_steps):
      f.write(str(weights[i]))
      f.write('\n')
    f.close
  print("--- %s seconds ---" % (time.time() - start_time))

make_dir = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
def get_out_path():
  #create dir and write path
  date = datetime.now().strftime("%H:%M:%S")
  out_path = (f"./results/scurves_data_{date}")
  make_dir("./results")
  make_dir(f"{out_path}")
  return out_path

def plot(path):
  slash_re = r'\/$'
  date_re = r'\d{2}:\d{2}:\d{2}'
  if re.search(slash_re, path):
    files = glob.glob(f"{path}*")
  else:
    files = glob.glob(f"{path}/*")
  colormap = plt.cm.get_cmap('viridis', len(files))
  for i,f in enumerate(files):
    print(f)
    weights = np.loadtxt(f, usecols=0);
    plt.plot(weights,color=colormap(i), alpha=0.7)
  plt.show()
  print("save figure? (y/n)")
  user_bool = input()
  if user_bool == 'y' or user_bool == 'Y':
    date = re.search(date_re, path).group(0)
    plt.savefig(f"./results/scurve_{date}.png",format='png',dpi=1200)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Call with task:\n'-g':generate scurve,\n'-p <path>': plot existing scurve with data from path")
    raise(IndexError)
  task = sys.argv[1]
  if task == '-g':
      gen()
  elif task == '-p':
      path = sys.argv[2]
      plot(path)
  else:
      print("can't do the task you've asked for")
      raise(NotImplementedError)
