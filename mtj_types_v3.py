import numpy as np

#============================== Private, global =====================================
# inject noise if device-to-device variation requested
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


#=========================== Parent class =========================================
class MTJ():
     # adds immutability to class. Only these values can be modified/created.
     # changes to the parameter list should go in here and the derivative valid_params
     __slots__ = ('Ms','Ki','TMR','Rp','J_she','a','b','tf','alpha','eta','d','t_pulse','t_relax',\
                  'theta','phi','phiHistory','thetaHistory','params_set_flag','sample_count',\
                  'mtj_type','valid_params','dflt_params')
     #================================================================================
     def __init__(self,mtj_type,dflt_params):
         self.valid_params = self.__slots__[0:12:1]
         self.phiHistory   = []
         self.thetaHistory = []
         self.sample_count = 0
         # using None value to check for proper initialization 
         self.phi   = None
         self.theta = None
         self.params_set_flag = None
         self.mtj_type = mtj_type
         self.dflt_params = dflt_params

     #================================================================================
     # check if all parameters have been set after each set attribute (wasteful)
     def __setattr__(self, name, value, check_flag=None):
         MTJ.__dict__[name].__set__(self, value)
         if(check_flag is None):
             self.check_parameters()
         else:
             pass

     #================================================================================
     # can call print(device) to list all parameters assigned
     def __str__(self):
         out_string = "\nDevice parameters:\n"
         for p in self.valid_params:
             try:
                 out_string += "\r" + str(p) + ": " + str(getattr(self,p)) + "\n"
             except(AttributeError):
                 out_string += "\r" + str(p) + ": " + " \n"
         return out_string

     #================================================================================
     def check_parameters(self):
         all_set = True
         for p in self.valid_params:
             if hasattr(self,p) is False:
                 all_set = False
         if all_set == True:
             #recusrive loop, exit with flag
             MTJ.__setattr__(self,"params_set_flag",True,check_flag=1)

     #================================================================================
     def set_mag_vector(self,phi,theta):
         self.phi   = phi
         self.theta = theta

     #================================================================================
     #set_vals can be called with True flag to use a default device setup
     #otherwise it takes individual device parameters passed in as in named arguemnts.
     #================================================================================
     def set_vals(self,dflt_flag = None,**params):
         #catch call with no arguments
         if params == {} and dflt_flag is None:
             print_key_error()
             raise(KeyError)
         #debug option with flag True: use known good device values
         elif ( dflt_flag == True or dflt_flag == False ) and params == {}:
             self.Ki    = draw_norm(self.dflt_params["Ki"], dflt_flag, 0.05)
             self.Rp    = draw_norm(self.dflt_params["Rp"], dflt_flag, 0.05)      # Magenetoresistance at parallel state, 8000 Ohm
             self.TMR   = draw_norm(self.dflt_params["TMR"], dflt_flag, 0.05)                # TMR ratio at V=0,120%  
             self.Ms    = self.dflt_params["Ms"]
             #=================
             self.J_she = self.dflt_params["J_she"]
             self.a     = self.dflt_params["a"]              # Width of the MTJ in m
             self.b     = self.dflt_params["b"]              # Length of the MTJ in m
             self.tf    = self.dflt_params["tf"]             # Thickness of the freelayer in m                           
             self.alpha = self.dflt_params["alpha"]          # Gilbert damping damping factor
             self.eta   = self.dflt_params["eta"]            # Spin hall angle
             self.d     = self.dflt_params["d"]              # Width,length and thichness of beta-W strip (heavy metal layer)
             self.t_pulse = self.dflt_params["t_pulse"]
             self.t_relax = self.dflt_params["t_relax"]
             self.params_set_flag = True

         elif dflt_flag is None:
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
#////////////////////////////////////////////////////////////////////////////////

# C-like enum. Should match Fortran
(SHE, SWrite, VCMA) = range(0, 3)

class SHE_MTJ_rng(MTJ):
    def __init__(self):
        # MTJ Parameters- This is experimental values from real STT-SOT p-MTJ%
        dflt_params = {"Ki" : 1.0056364e-3,
                    "Rp" : 5e3,
                    "TMR" : 1.2,
                    "Ms" : 1.2e6,
                    "J_she" : 5e11,
                    "a"  : 50e-9,
                    "b"  : 50e-9,
                    "tf" : 1.1e-9,
                    "alpha" :0.03,
                    "eta" : 0.3,
                    "d"  : 3e-9,
                    "t_pulse" : 10e-9,
                    "t_relax" : 15e-9}
        super().__init__(SHE,dflt_params)

class SWrite_MTJ_rng(MTJ):
    def __init__(self):
        dflt_params = {"Ki" : 1.0056364e-3,
                    "Rp" : 5e3,
                    "TMR" : 1.2,
                    "Ms" : 1.2e6,
                    "J_she" : 5e11,
                    "a"  : 50e-9,
                    "b"  : 50e-9,
                    "tf" : 1.1e-9,
                    "alpha" :0.03,
                    "eta" : 0.3,
                    "d"  : 3e-9,
                    "t_pulse" : 10e-9,
                    "t_relax" : 15e-9}
        super().__init__(SWrite,dflt_params)
