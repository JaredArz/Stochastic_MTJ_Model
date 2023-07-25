import sys
# import functions to test directly
sys.path.append('../')
sys.path.append('../fortran_source')

#files to test, should work well together.
import single_sample as ss
import mtj_types_Ki as mtj
import numpy as np
def sampling_wrapper(dev,Jstt,view_mag_flag):
        energy, bit, theta_end, phi_end = ss.single_sample.pulse_then_relax(Jstt,\
                dev.J_she,dev.theta,dev.phi,dev.Ki,dev.TMR,dev.t_pulse,\
                dev.a,dev.b,dev.tf,dev.alpha,dev.Ms,dev.eta,dev.d,\
                view_mag_flag)
        dev.theta = theta_end
        dev.phi   = phi_end
        dev.thetaHistory.append(theta_end)
        dev.phiHistory.append(phi_end)
        return bit,energy

num_iter = 100
fsum_t = 0
fsum_p = 0
#Using known good values
th_init =  np.pi/2
ph_init = np.pi/2
alpha   = 0.03            # damping constant
Ms   = 1.2e6        # saturation magnetization
TMR   = 1.2                      # tunneling magnetoresistance ratio
Ki   = 0.9056364e-3   # anistrophy energy    
Rp   = 8e3          # parallel resistance
d   = 3e-9                                     # free layer diameter
tf   = 1.1e-9                                    # free layer thickness
eta   = 0.3                # spin hall angle
t_pulse = 10e-9     # pulse duration
J_she   = 1e-3 # current density
Jappl = 1e-3

dev = mtj.SHE_MTJ_rng(th_init,ph_init,Ms,Ki,1)

dev.TMR = TMR
dev.alpha = alpha
dev.Ki = Ki
dev.Ms = Ms
dev.Rp = Rp
dev.a = 50e-9 
dev.b = 50e-9
dev.tf = tf
dev.eta = eta
dev.t_pulse = t_pulse
dev.J_she = J_she
dev.d=d

for i in range(num_iter):
    bit,energy = sampling_wrapper(dev,Jappl,0)
    fsum_t += dev.theta
    fsum_p += dev.phi

print(f"            debug         ")
print(f"final theta: {fsum_t/num_iter:.4f}\n\
final phi:   {fsum_p/num_iter:.4f}")
