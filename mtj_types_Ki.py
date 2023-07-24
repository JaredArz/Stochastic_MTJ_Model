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

    def __init__(self,init_th,ph_init,init_Ms,init_Ki,var):
        if init_th == 0:
            print('Init theta cannot be 0, defaulted to pi/100')
            r_init = np.pi/100
            self.theta = r_init
        else:
            self.theta = init_th
        self.phi = ph_init
        self.J_she = 5e11
        self.t_pulse = 10e-9
        self.bitstr = []

        # MTJ Parameters- This is experimental values from real STT-SOT p-MTJ%
        self.a = 50e-9                       # Width of the MTJ in m
        self.b = 50e-9                       # Length of the MTJ in m
        self.tf = 1.1e-9                     # Thickness of the freelayer in m                           
        self.alpha = 0.03                    # Gilbert damping damping factor
        #self.Ms = 1.2e6                      # Saturation magnetization in A/m
        self.Ms = init_Ms
        self.TMR = draw_norm(1.5,var,0.05)                       # TMR ratio at V=0,120%  
        #self.TMR = 1.2
        self.Rp = draw_norm(self.RA/self.A,var,0.05)                        # Magenetoresistance at parallel state
        #self.Rp = 8e3
        #self.Ki = draw_norm(self.Eb/self.A,var,0.05)               # The anisotropy energy in J/m2
        #self.Ki = 0.906366e-3
        self.Ki = init_Ki
        self.eta = 0.3                       # Spin hall angle
        self.d = 3e-9                        # Width,length and thichness of beta-W strip (heavy metal layer)

        '''deprecated. Nx,Ny,Nz now declared in ./fortran_source/she_mtj_params.f90 
        # Approximate demagnetization field; this assumes a square plate
        #self.Nx = 2*self.tf/(np.pi*self.a)
        #self.Ny = 2*self.tf/(np.pi*self.b)
        #self.Nz = 1-(self.Nx+self.Ny)
        # Demagnetization field; This is calculated using MuMax for 34nm eCD; probably an underestimate?
        # self.Nx = 0.0109358
        # self.Ny = 0.0109358
        # self.Nz = 1-(self.Nx + self.Ny)
        '''
        self.phiHistory = []
        self.thetaHistory = []

    def cont_sample(self):
        pass
