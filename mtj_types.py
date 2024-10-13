import numpy as np
import json

from pathlib import Path

main_dir = Path(__file__).parent
parameters_file = main_dir / "mtj_parameters.json"

# C-like enum. Should match Fortran
(SHE, SWrite, VCMA) = range(0, 3)


# ============================= Parent class =========================================
class MTJ:
    def __init__(self, mtj_type, dflt_params, dflt_m, heating_capable):
        self.valid_params = [param for param, _ in dflt_params.items()]
        self.dflt_params = dflt_params
        self.dflt_m = dflt_m
        self.mtj_type = mtj_type
        self.phiHistory = []
        self.thetaHistory = []
        self.tempHistory = []
        self.sample_count = 0
        self.heating_enabled = 0
        self.heating_capable = heating_capable

    # ================================================================================
    # can use built-in function 'print' to print(dev) and list all parameters
    def __str__(self):
        out_string = "\nDevice parameters:\n"
        for p in self.valid_params:
            try:
                out_string += "\r" + str(p) + ": " + str(getattr(self, p)) + "\n"
            except AttributeError:
                out_string += "\r" + str(p) + ": " + " \n"
        out_string += "\rHeating enabled: " + str(bool(self.heating_enabled)) + " \n"
        return out_string

    # ================================================================================
    def set_mag_vector(self, phi=None, theta=None):
        if phi is None and theta is None:
            self.phi = self.dflt_m["phi"]
            self.theta = self.dflt_m["theta"]
        else:
            self.phi = phi
            self.theta = theta
        return

    def init(self):
        self.set_mag_vector()
        self.set_vals()
        return

    def enable_heating(self):
        self.heating_enabled = 1
        return

    def disable_heating(self):
        self.heating_enabled = 0
        return

    # ================================================================================
    # A list of valid parameters determined by the subclass is used to check whether the arguments
    # passed to this function are valid.
    # Runtime errors at the fortran interface will catch any parameters that are not yet set at
    # sampling time.
    def set_vals(self, **params):
        # no args => use default param
        if params == {}:
            for key, val in self.dflt_params.items():
                self.__setattr__(key, val)
        # try and set custom param values
        elif params != {}:
            try:
                for key, val in params.items():
                    if key not in self.valid_params:
                        raise (KeyError)
                    self.__setattr__(key, val)
            except KeyError:
                self.print_key_error()
                raise
        return

    def print_init_error(self):
        print("--------------------------------*--*-----------------------------------")
        print("Check that the magnetization vector was initalized.")
        print(
            "Otherwise, one or more parameter values were passed incorrectly, or not at all before sampling."
        )
        self.print_expected_params()
        print("--------------------------------*--*-----------------------------------")

    def print_key_error(self):
        print("--------------------------------*--*-----------------------------------")
        print("An attempt was made to assign/access an invalid parameter.")
        self.print_expected_params()
        print("--------------------------------*--*-----------------------------------")

    def print_expected_params(self):
        print("Expecting for all MTJ types:")
        print("Ms, Ki, TMR, Rp,a, b, tf, alpha, eta, d, t_pulse, t_relax, T,")
        print("and for ", end="")
        if self.mtj_type == SHE:
            print("SHE: J_she, Hy ( optional )")
        elif self.mtj_type == SWrite:
            print("SWrite: J_reset, t_reset, H_reset (optional)")
        elif self.mtj_type == VCMA:
            print("VCMA: v_pulse")

    def print_device_not_found(self):
        print("-------------*--*----------------")
        print("Device subtype not found.")
        print("Subtypes available:")
        print("SHE: None (default UTA)")
        print("SWrite: UTA, NYU")
        print("VCMA: None (default UTA)")
        print("--------------*--*---------------")


# ================================================================================
# ////////////////////////////////////////////////////////////////////////////////


class SHE_MTJ_rng(MTJ):
    def __init__(self):
        dflt_m = {
            "theta": np.random.normal(0, np.pi / 4),
            "phi": np.random.rand() * 2 * np.pi,
        }
        heating_capable = 0
        with parameters_file.open("r") as f:
            dflt_params = (json.load(f))["SOT"]
        super().__init__(SHE, dflt_params, dflt_m, heating_capable)


class SWrite_MTJ_rng(MTJ):
    def __init__(self, flavor):
        dflt_m = {
            "theta": np.random.normal(np.pi, np.pi / 4),
            "phi": np.random.rand() * 2 * np.pi,
        }
        if flavor == "UTA":
            heating_capable = 0
            with parameters_file.open("r") as f:
                dflt_params = (json.load(f))["SWRITE"]["UTA"]
        elif flavor == "NYU":
            heating_capable = 1
            with parameters_file.open("r") as f:
                dflt_params = (json.load(f))["SWRITE"]["NYU"]
        else:
            self.print_device_not_found()
            exit()
        super().__init__(SWrite, dflt_params, dflt_m, heating_capable)


class VCMA_MTJ_rng(MTJ):
    def __init__(self):
        dflt_m = {
            "theta": np.random.normal(0, np.pi / 4),
            "phi": np.random.rand() * 2 * np.pi,
        }
        heating_capable = 0
        with parameters_file.open("r") as f:
            dflt_params = (json.load(f))["VCMA"]
        super().__init__(VCMA, dflt_params, dflt_m, heating_capable)
