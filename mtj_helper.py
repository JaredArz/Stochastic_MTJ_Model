import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.style as style
import scienceplots
from datetime import datetime

from interface_funcs import mtj_sample

def draw_norm(x,psig):
    return (x*np.random.normal(1,psig))

def draw_const(x,csig):
    return (x+np.random.normal(-csig,csig))

# Device-to-device variation and cycle-to-cycle variation
# can be modeled simply with this function.
# =======
# Takes a device, device parameter, and a percent deviation
# (which defines the standard deviation of a sampled
# gaussian distribution around the current parameter value).
# =======
# Returns the modified device.
def vary_param(dev, param, stddev):
    current_val = dev.__getattribute__(param)
    updated_val = draw_norm(current_val, stddev)
    dev.__setattr__(param,updated_val)
    return dev

def plot_init():
    #plt.rc('text', usetex=True)
    #plt.style.use(['science','ieee'])
    fig, ax = plt.subplots()
    #fig.tight_layout()
    return fig,ax

def prompt_show():
    valid_user_input = False
    while(not valid_user_input):
        print("Show figure? y/n")
        user_input = input()
        if user_input == 'y' or user_input == 'Y':
            plt.show()
            valid_user_input = True
        elif user_input == 'n' or user_input == 'N':
            valid_user_input = True
        else:
            print("invalid input.")

def prompt_save_svg(fig,path):
    valid_user_input = False
    while(not valid_user_input):
        print("Save figure? y/n")
        user_input = input()
        if user_input == 'y' or user_input == 'Y':
            fig.savefig(path, format='svg', dpi=1200)
            valid_user_input = True
        elif user_input == 'n' or user_input == 'N':
            valid_user_input = True
        else:
            print("invalid input.")

def avg_weight_across_samples(dev, V, samples_to_avg) -> float:
    sum_p = np.sum( [ (mtj_sample(dev, V),) for _ in range(samples_to_avg)] )
    return sum_p/samples_to_avg

def find_idx_at_nearest(vec, val) -> int:
    vector_difference = np.abs( np.asarray(vec) - val )
    return vector_difference.argmin()

def gamma_pdf(g1, g2, nrange) -> list:
  # Build an analytical gamma probability density function (PDF)

  # g1 corresponds to alpha in gamma distribution definitinon
  # g2 corresponds to beta in gamma distribution, or lmda in previous work here
  # g1 must be an integer for this formula to work. if non-integer g1 are desired, factorial function should become gamma function
  xxis = []
  pdf = []
  for j in range(nrange):
    gval = pow(j,g1-1)*pow(g2,g1)*np.exp(-g2*j)/factorial(g1-1)
    xxis.append(j)
    pdf.append(gval)

  # Normalize exponential distribution
  pdfsum = 0
  for j in range(nrange):
    pdfsum += pdf[j]

  pdf = pdf/pdfsum

  return pdf
