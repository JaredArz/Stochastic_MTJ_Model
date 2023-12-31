import matplotlib.pyplot as plt
import matplotlib.style as style
import scienceplots

def plot_init():
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.style.use(['science','ieee'])
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
