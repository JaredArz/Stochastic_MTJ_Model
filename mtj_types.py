import numpy as np

# constants
uB = 9.274e-24
h_bar = 1.054e-34          
u0 = np.pi*4e-7               
e = 1.6e-19                
kb = 1.38e-23   
gamma = 2*u0*uB/h_bar           # Gyromagnetic ratio in m/As
gamma_b = gamma/u0

def draw_norm(x,var,psig):
    if var:
        return x*np.random.normal(1,psig,1)
    else:
        return x

def draw_const(x,var,csig):
    if var:
        return x + np.random.normal(-csig,csig,1)
    else:
        return x
    
class SHE_MTJ_rng():

    def __init__(self,init_theta,phi_init,var):
        if init_theta == 0:
            print('Init theta cannot be 0, defaulted to pi/100')
            _ = np.pi/100
            self.theta = _
        else:
            self.theta = init_theta
        self.phi = phi_init

        #FIXME: check multiply
        self.Ki  = draw_norm(0.9056364e-3,var,0.05)    # The anisotropy energy in J/m2
        self.TMR = draw_norm(1.2,var,0.05)             # TMR ratio at V=0,120%  
        self.Rp  = draw_norm(8e3,var,0.05)             # Magenetoresistance at parallel state, 8000 Ohm
    
    def cont_sample(self):
        pass
