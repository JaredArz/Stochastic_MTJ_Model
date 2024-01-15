
'''
def error_function(args):
    Ms_factor, Ki_factor = args
    out_path = get_out_path()
    dev = SWrite_MTJ_rng()
    dev.set_mag_vector()
    dev.set_vals(0)
    dev.set_vals(a=40e-9, b=40e-9, TMR = 1.24, tf = 2.6e-9, Rp = 2530, Ki=Ki_factor*4.1128e-4, Ms=Ms_factor*4.73077e5)
    gen_fig1_data(dev, out_path)
    file = glob.glob(out_path + "/*voltage_sweep*")[0]
    f_data = np.load(file)
    sim_weights = np.flip(np.array(f_data["weights"]))

    #fig1aLR_x = [t*1e9 for t in (np.loadtxt('./fig1aLR.txt',usecols=0))]
    #fig1aLR_y = np.loadtxt('./fig1aLR.txt',usecols=1)
    #fig1bLR300_x = np.loadtxt('./fig1bLR300.txt',usecols=0)
    fig1bLR300_y = np.loadtxt('./fig1bLR300.txt',usecols=1)

    #pulse_durations = np.linspace(0, 3.2e-9, 28) #250
    #voltages = np.linspace(-0.97919268, -0.43084478, 28) #250
    #print(sim_weights)
    #print(fig1bLR300_y)
    #input()

    #print(f"voltage sim x axis: {voltages}")
    #input(f"exp x axis: {fig1bLR300_x}")
    #print(np.nan_to_num(np.average(np.abs( sim_weights - fig1bLR300_y ) / fig1bLR300_y)))
    #print(np.average(np.nan_to_num(np.abs( sim_weights - fig1bLR300_y ) / fig1bLR300_y)))
    #input()
    return np.average(np.nan_to_num(np.abs( sim_weights - fig1bLR300_y )))

def optimize_param():
    #initial_guess = [0.5, 2] #ms, ki
    initial_guess = [0.5, 2] #ms, ki
    result = optimize.minimize(error_function, initial_guess, tol=0.3)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)
    return
'''
