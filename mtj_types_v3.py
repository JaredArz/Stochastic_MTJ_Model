import numpy as np

#============================== Private ========================================
# inject noise if device-to-device variation requested
def draw_norm(x,var,psig):
    return (x if not var else(x*np.random.normal(1,psig)))

def draw_const(x,var,csig):
    return (x if not var else(x+np.random.normal(-csig,csig)))

#================================================================================
#////////////////////////////////////////////////////////////////////////////////

# C-like enum. Should match Fortran
(SHE, SWrite, VCMA) = range(0, 3)

#============================= Parent class =========================================
class MTJ():
     def __init__(self,mtj_type,dflt_params,dflt_noise,dflt_m):
         shared_params = ('Ms','Ki','TMR','Rp','a','b','tf','alpha','eta','d','t_pulse','t_relax')
         valid_params = list(shared_params)
         # add parameters unique to mtj type into valid_params
         for key, _ in dflt_params.items():
             if key not in valid_params:
                 valid_params.append(key)
         self.valid_params = valid_params
         self.dflt_params = dflt_params
         self.dflt_noise  = dflt_noise
         self.dflt_m      = dflt_m
         self.mtj_type    = mtj_type
         self.phiHistory   = []
         self.thetaHistory = []
         self.sample_count = 0

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
     def set_mag_vector(self,phi,theta):
         if phi == None and theta == None:
             self.phi = dflt_m["phi"]
             self.theta = dflt_m["theta"]
         else:
             self.phi   = phi
             self.theta = theta

     #================================================================================
     # A list of valid parameters determined by the subclass is used to check whether the arguments
     # passed to this function are valid.
     # Runtime errors at the fortran interface will catch any parameters that are not yet set at 
     # sampling time.
     def set_vals(self,dflt_flag = None,**params):
         #catch call with no arguments
         if params == {} and dflt_flag is None:
             raise(ValueError)
         #debug option with flag True: use known good device values
         elif ( dflt_flag == True or dflt_flag == False ) and params == {}:
             for key, val in self.dflt_params.items():
                try:
                    val = draw_norm(val, dflt_flag, self.dflt_noise[key])
                except(KeyError):
                    pass
                finally:
                    self.__setattr__(key, val)
             self.params_set_flag = True

         elif dflt_flag is None:
             try:
                 for key, val in params.items():
                     if key not in self.valid_params:
                         raise(KeyError)
                     self.__setattr__(key, val)
             except(KeyError):
                 self.print_key_error()
                 raise
         #catch anything else, just in case
         else:
             print_key_error()
             raise(KeyError)
     def print_init_error(self):
          print("--------------------------------*--*-----------------------------------")
          print("Check that the magnetization vector was initalized.")
          print("Otherwise, one or more parameter values were passed incorrectly, or not at all before sampling.")
          self.print_expected_params()
          print("--------------------------------*--*-----------------------------------")

     def print_key_error(self):
         print("--------------------------------*--*-----------------------------------")
         print("An attempt was made to assign an invalid parameter to this device.")
         self.print_expected_params()
         print("--------------------------------*--*-----------------------------------")

     def print_expected_params(self):
         print("Expecting for all MTJ types:")
         print("Ms, Ki, TMR, Rp,a, b, tf, alpha, eta, d, t_pulse, t_relax,")
         print("and for ",end="")
         if self.mtj_type == SHE:
             print("SHE: J_she, Hy")
         elif self.mtj_type == SWrite:
             print("SWrite: J_reset, H_reset, H_appl, t_reset")
         elif self.mtj_type == VCMA:
             print("VCMA: v_pulse")
#================================================================================
#////////////////////////////////////////////////////////////////////////////////

class SHE_MTJ_rng(MTJ):
    def __init__(self):
        # MTJ Parameters- This is experimental values from real STT-SOT p-MTJ%
        dflt_m = {"theta" : np.pi/100,
                  "phi"   : np.random.rand()*2*np.pi}
        dflt_noise = {"Ki"  : 0.05,
                      "Rp"  : 0.05,
                      "TMR" : 0.05,}
        dflt_params = {"Ki" : 1.0056364e-3,"Rp" : 5e3,
                       "TMR" : 1.2,    "Ms" : 1.2e6,
                       "J_she" : 5e11, "a"  : 50e-9,
                       "b"  : 50e-9,   "tf" : 1.1e-9,
                       "alpha" :0.03,  "eta" : 0.3,
                       "d"  : 3e-9,    "t_pulse" : 10e-9,
                       "t_relax" : 15e-9, "Hy": 0}
        super().__init__(SHE,dflt_params,dflt_noise,dflt_m)

class SWrite_MTJ_rng(MTJ):
    def __init__(self):
        dflt_m = {"theta"  : 99*np.pi/100,
                  "phi"    : np.random.rand()*2*np.pi}
        dflt_noise = {"Ki"  : 0.05,
                      "Rp"  : 0.05,
                      "TMR" : 0.05,}
        dflt_params = {"Ki" : 1.0056364e-3,"Rp" : 5e3,
                       "TMR": 1.2,         "Ms" : 1.2e6,
                       "J_reset": 5e11,    "H_reset": 0,
                       "H_appl":0,         "a"  : 50e-9,
                       "b"  : 50e-9,       "tf" : 1.1e-9,
                       "alpha" :0.03,      "eta" : 0.3,
                       "d"  : 3e-9,        "t_pulse" : 1e-9,
                       "t_relax" : 10e-9,  "t_reset" : 10e-9 }
        super().__init__(SWrite,dflt_params,dflt_noise,dflt_m)

class VCMA_MTJ_rng(MTJ):
    def __init__(self):
        dflt_m = {"theta" : np.pi/100,
                  "phi"   : np.random.rand()*2*np.pi}
        dflt_noise = {"Ki"  : 0.05,
                      "Rp"  : 0.05,
                      "TMR" : 0.05,}
        dflt_params = {"Ki" : 1.0056364e-3,"Rp" : 5e3,
                       "TMR" : 1.2,    "Ms" : 1.2e6,
                       "v_pulse" : 1.5, "a"  : 50e-9,
                       "b"  : 50e-9,   "tf" : 1.1e-9,
                       "alpha" :0.03,  "eta" : 0.3,
                       "d"  : 3e-9,    "t_pulse" : 50e-9,
                       "t_relax" : 15e-9}
        super().__init__(VCMA,dflt_params,dflt_noise,dflt_m)
