import numpy as np

#========================== helper functions ==================================
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

def print_key_error():
     print("One or more parameter values were passed incorrectly or not at all.")
     print("Use named parameters, expecting: Ms,Ki,TMR,Rp,J_she,a,b,tf,alpha,eta,d.")
     print("--------------------------------*--*-----------------------------------")
#================================================================================
#////////////////////////////////////////////////////////////////////////////////



# this list, along with __slots__ dictate what the class is expecting
# changes to the parameter list should go in here, __slots__ and to the debug option in set_vals
all_params = ('Ms','Ki','TMR','Rp','J_she','a','b','tf','alpha','eta','d')

class SHE_MTJ_rng():
     #   adds immutability to class. Only these values can be modified/created.
     __slots__ = ('theta','phi','Ms','Ki','TMR','Rp','J_she','a','b','tf','alpha','eta','d',
                  'phiHistory','thetaHistory','dd_flag','params_set_flag')

     #================================================================================
     # initialize device with a dev-to-dev variation flag
     def __init__(self,dd_flag):
         self.dd_flag=dd_flag
         self.phiHistory   = []
         self.thetaHistory = []
         # use None value to check for mag initialization 
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
     def set_vals(self,debug_flag=None,**params):
         #catch call with no arguments
         if params == {} and debug_flag is None:
             print_key_error()
             raise(KeyError)
         #debug option with flag True: use known good device values
         elif debug_flag == True and params == {}:
             # MTJ Parameters- This is experimental values from real STT-SOT p-MTJ%
             self.J_she = 5e11
             self.Ms = 0.4e6
             self.Ki = 0.00014759392802570008
             self.TMR   = draw_norm(1.5,self.dd_flag,0.05)  # TMR ratio at V=0,120%  
             self.Rp    = 3861.20994613      # Magenetoresistance at parallel state, 8000 Ohm
             self.a     = 50e-9              # Width of the MTJ in m
             self.b     = 50e-9              # Length of the MTJ in m
             self.tf    = 1.1e-9             # Thickness of the freelayer in m                           
             self.alpha = 0.03               # Gilbert damping damping factor
             self.eta   = 0.3                # Spin hall angle
             self.d     = 3e-9               # Width,length and thichness of beta-W strip (heavy metal layer)
             self.params_set_flag = True
         #normal usage, process args and set flag if device fully initialized
         elif debug_flag is None:
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
