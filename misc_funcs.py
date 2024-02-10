import matplotlib.pyplot as plt
import numpy as np

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

def find_idx_at_nearest(vec, val) -> int:
    vector_difference = np.abs( np.asarray(vec) - val )
    return vector_difference.argmin()
