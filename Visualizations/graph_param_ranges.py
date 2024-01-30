import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

s1 = 1
s2 = 5

# Parameter values
alpha = 0.01
Ki = 0.0002
Ms = 300000.0
Rp = 500.0
TMR = 0.3
eta = 0.26508163213729863
J_she = 1000000000000.0
t_pulse = 3.790034800767898e-08

# alpha = 0.0433191192150116
# Ki = 0.0002
# Ms = 300000.0
# Rp = 500.0
# TMR = 0.3
# eta = 0.7890509426593781
# J_she = 137515515685.08148
# t_pulse = 2.2646484002470972e-08

# Parameter ranges
alpha_range = [0.01, 0.1]
Ki_range = [0.2e-3, 1e-3]
Ms_range = [0.3e6, 2e6]
Rp_range = [500, 50000]
TMR_range = [0.3, 6]
eta_range = [0.1, 0.8]
J_she_range = [0.01e12, 1e12]
t_pulse_range = [0.5e-9, 75e-9]
t_relax_range = [0.5e-9, 75e-9]


def remap(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def get_info(val):
  if val == 8:
    start_label = '{:.1E}'.format(alpha_range[0])
    end_label = '{:.1E}'.format(alpha_range[1])
    val_label = '{:.1E}'.format(alpha)
    coord = remap(alpha, alpha_range[0], alpha_range[1], s1, s2)
    coord = [coord, val]
  elif val == 7:
    start_label = '{:.1E}'.format(Ki_range[0])
    end_label = '{:.1E}'.format(Ki_range[1])
    val_label = '{:.1E}'.format(Ki)
    coord = remap(Ki, Ki_range[0], Ki_range[1], s1, s2)
    coord = [coord, val]
  elif val == 6:
    start_label = '{:.1E}'.format(Ms_range[0])
    end_label = '{:.1E}'.format(Ms_range[1])
    val_label = '{:.1E}'.format(Ms)
    coord = remap(Ms, Ms_range[0], Ms_range[1], s1, s2)
    coord = [coord, val]
  elif val == 5:
    start_label = '{:.1E}'.format(Rp_range[0])
    end_label = '{:.1E}'.format(Rp_range[1])
    val_label = '{:.1E}'.format(Rp)
    coord = remap(Rp, Rp_range[0], Rp_range[1], s1, s2)
    coord = [coord, val]
  elif val == 4:
    start_label = '{:.1E}'.format(TMR_range[0])
    end_label = '{:.1E}'.format(TMR_range[1])
    val_label = '{:.1E}'.format(TMR)
    coord = remap(TMR, TMR_range[0], TMR_range[1], s1, s2)
    coord = [coord, val]
  elif val == 3:
    start_label = '{:.1E}'.format(eta_range[0])
    end_label = '{:.1E}'.format(eta_range[1])
    val_label = '{:.1E}'.format(eta)
    coord = remap(eta, eta_range[0], eta_range[1], s1, s2)
    coord = [coord, val]
  elif val == 2:
    start_label = '{:.1E}'.format(J_she_range[0])
    end_label = '{:.1E}'.format(J_she_range[1])
    val_label = '{:.1E}'.format(J_she)
    coord = remap(J_she, J_she_range[0], J_she_range[1], s1, s2)
    coord = [coord, val]
  elif val == 1:
    start_label = '{:.1E}'.format(t_pulse_range[0])
    end_label = '{:.1E}'.format(t_pulse_range[1])
    val_label = '{:.1E}'.format(t_pulse)
    coord = remap(t_pulse, t_pulse_range[0], t_pulse_range[1], s1, s2)
    coord = [coord, val]
  else:
    raise KeyError('Invalid val')
  
  return start_label, end_label, val_label, coord


def graph():
  fig, ax = plt.subplots()
  ax.axis(xmin=s1-1, xmax=s2+1, ymin=0, ymax=9)
  ax.get_xaxis().set_visible(False)
  ax.set_yticks((1, 2, 3, 4, 5, 6, 7, 8))
  ax.set_yticklabels(('t_pulse', 'J_SHE', 'η', 'TMR', 'R_p', 'M_s', 'K_i', 'α'), weight='bold')

  for i in range(1,9):
    start_label, end_label, val_label, coord = get_info(i)
    x = [s1, s2]
    y = [i, i]

    ax.plot(x, y, color='#4B4B4B', zorder=0)
    ax.scatter(x, y, s=8, color='#4B4B4B', zorder=0)
    ax.scatter(coord[0], coord[1], s=50, color='#FF8200', zorder=1)

    ax.annotate(start_label, (x[0], y[0]), xytext=(x[0]-0.1, y[0]), size=14, ha='right', va='center')
    ax.annotate(end_label, (x[1], y[1]), xytext=(x[1]+0.1, y[1]), size=14, ha='left', va='center')
    ax.annotate(val_label, (coord[0], coord[1]), xytext=(coord[0], coord[1]+0.1), size=18, ha='center', va='bottom')
  
  plt.show()


if __name__ == '__main__':
  graph()