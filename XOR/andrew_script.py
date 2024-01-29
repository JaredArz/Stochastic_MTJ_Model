import sys
sys.path.append("../")
import helper_funcs as helper
import XOR_funcs as funcs
import helper_funcs
import numpy as np
from scipy.stats import chi2

word_size   = 8
record_size = 100000
dof = 256
prob = [2**(-1*word_size) for _ in range(dof)]

class data_set():
        def __init__(self, fname):
            self.fname = fname
            self.data  = np.load(fname)
            self.uniformity = funcs.get_uniformity(self.data, word_size, record_size)
            self.chisq = funcs.compute_chi_squared(self.uniformity, word_size, record_size)
            self.p_val = chi2.sf(self.chisq, 256)
            self.x = self.uniformity / record_size

if (len(sys.argv)) != 2:
    print("Error: pass file to get data from.")

file = sys.argv[1]
user_data = data_set(file)
print(f"chisq: {user_data.chisq}\np value: {user_data.p_val}")
