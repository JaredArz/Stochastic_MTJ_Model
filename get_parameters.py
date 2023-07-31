import numpy as np

alpha_vals   = [0.01, 0.03, 0.05, 0.07, 0.1]            # damping constant
Ki_vals      = [0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1e-3]   # anistrophy energy    
Ms_vals      = [0.3e6, 0.7e6, 1.2e6, 1.6e6, 2e6]        # saturation magnetization
Rp_vals      = [500, 1000, 5000, 25000, 50000]          # parallel resistance
TMR_vals     = [0.3, 0.5, 2, 4, 6]                      # tunneling magnetoresistance ratio
d_vals       = [50]                                     # free layer diameter
tf_vals      = [1.1]                                    # free layer thickness
eta_vals     = [0.1, 0.2, 0.4, 0.6, 0.8]                # spin hall angle
t_pulse_vals = [0.5e-9, 1e-9, 10e-9, 50e-9, 75e-9]      # pulse duration
J_she_vals   = [0.01e12, 0.1e12, 0.25e12, 0.5e12, 1e12] # current density

parameters = []

for alpha in alpha_vals:
  for Ki in Ki_vals:
    for Ms in Ms_vals:
      for Rp in Rp_vals:
        for TMR in TMR_vals:
          for d in d_vals:
            for tf in tf_vals:
              for eta in eta_vals:
                for t_pulse in t_pulse_vals:
                  for J_she in J_she_vals:
                    p = [alpha, Ki, Ms, Rp, TMR, d, tf, eta, t_pulse, J_she]
                    parameters.append(p)

path = "parameter_config.npy"
np.save(path, np.array(parameters))
print(np.shape(parameters))
