import numpy as np

# Private functions to inject noise if device-to-device variation requested
def draw_norm(x,var,psig):
    return (x if not var else(x*np.random.normal(1,psig)))

def draw_const(x,var,csig):
    return (x if not var else(x+np.random.normal(-csig,csig)))

def print_key_error():
     print("One or more parameter values were passed incorrectly or not at all.")
     print("Use named parameters, expecting the following to be set before use:")
     print("Ms, Ki, TMR, Rp, J_she, a, b, tf, alpha, eta, d, t_pulse, t_relax")
     print("--------------------------------*--*-----------------------------------")

#================================================================================
#////////////////////////////////////////////////////////////////////////////////

# this list, along with __slots__ dictate what the class is expecting
# changes to the parameter list should go in here, __slots__ and to the debug option in set_vals
all_params = ('Ms','Ki','TMR','Rp','J_she','a','b','tf','alpha','eta','d','t_pulse','t_relax')

class SHE_MTJ_rng():
     #   adds immutability to class. Only these values can be modified/created.
     __slots__ = ('theta','phi','Ms','Ki','TMR','Rp','J_she','a','b','tf','alpha','eta','d','t_pulse','t_relax',\
                    'phiHistory','thetaHistory','dd_var_flag','params_set_flag','sample_count')
     #================================================================================
     def __init__(self):
         self.phiHistory   = []
         self.thetaHistory = []
         self.sample_count = 0
         # using None value to check for proper initialization 
         self.phi   = None
         self.theta = None
         self.params_set_flag = None

     #================================================================================
     # check if all parameters have been set after each set attribute (wasteful)
     def __setattr__(self, name, value, check_flag=None):
         SHE_MTJ_rng.__dict__[name].__set__(self, value)
         if(check_flag is None):
             self.check_parameters()
         else:
             pass

     #================================================================================
     # can call print(device) to list all parameters assigned
     def __str__(self):
         out_string = "\nDevice parameters:\n"
         for p in all_params:
             try:
                 out_string += "\r" + str(p) + ": " + str(getattr(self,p)) + "\n"
             except(AttributeError):
                 out_string += "\r" + str(p) + ": " + " \n"
         return out_string

     #================================================================================
     def check_parameters(self):
         all_set = True
         for p in all_params:
             if hasattr(self,p) is False:
                 all_set = False
         if all_set == True:
             #recusrive loop, exit with flag
             SHE_MTJ_rng.__setattr__(self,"params_set_flag",True,check_flag=1)

     #================================================================================
     def set_mag_vector(self,phi,theta):
         self.phi   = phi
         self.theta = theta

     #================================================================================
     #set_vals can be called with True flag to use a default device setup
     #otherwise it takes individual device parameters passed in as in named arguemnts.
     #================================================================================
     def set_vals(self,default_flag=None,**params):
         #catch call with no arguments
         if params == {} and default_flag is None:
             print_key_error()
             raise(KeyError)
         #debug option with flag True: use known good device values
         elif ( default_flag == True or default_flag == False ) and params == {}:
             # MTJ Parameters- This is experimental values from real STT-SOT p-MTJ%
             self.Ki    = draw_norm(1.0056364e-3, default_flag, 0.05)
             self.Rp    = draw_norm(5e3, default_flag, 0.05)      # Magenetoresistance at parallel state, 8000 Ohm
             self.TMR   = draw_norm(1.2, default_flag, 0.05)                # TMR ratio at V=0,120%  
             self.Ms    = 1.2e6
             #=================
             self.J_she = 5e11
             self.a     = 50e-9              # Width of the MTJ in m
             self.b     = 50e-9              # Length of the MTJ in m
             self.tf    = 1.1e-9             # Thickness of the freelayer in m                           
             self.alpha = 0.03               # Gilbert damping damping factor
             self.eta   = 0.3                # Spin hall angle
             self.d     = 3e-9               # Width,length and thichness of beta-W strip (heavy metal layer)
             self.t_pulse = 10e-9
             self.t_relax = 15e-9
             self.params_set_flag = True

         elif default_flag is None:
             try:
                 for param_key, param_val in params.items():
                     self.__setattr__(param_key, param_val, 1)
             except(KeyError):
                 print_key_error()
                 raise
             finally:
                 self.check_parameters()
         #catch anything else, just in case
         else:
             print_key_error()
             raise(KeyError)
     #================================================================================
