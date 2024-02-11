# Stochastic MTJ device models
### contact: jared.arzate@utexas.edu
##  Device models included:
### - SOT MTJ
### - VCMA MTJ
### - STT Stochastic Write MTJ

## Working:
### - [X] Serial
### - [ ] Parallel
### - [X] Thread safe

## Usage:
First, navigate into the fortran_source directory and compile with:
```
cd ./fortran_source
make
```
The fortran compiles into a cpython binary which can be imported as a module from python.
This binary does the heavy lifting, computing the magnetization dynamics of a stochastic SOT-driven MTJ via a modified LLG equation.

`interface_funcs.py` has a user function `mtj_sample(device, J_stt,...)` which will call this binary and handle the language interface.

`mtj_sample(device, J_stt,...)` simultes the pulsing and relaxing of
a *_MTJ_rng device (class described below) passed in as the first argument.
This device class should be intialized with device parameters and an initial magnetization vector.

The other arguments are as follows:
- Jstt:     spin-transfer torque current to apply.

(enter the following as named arguments)
- view_mag_flag: enables/disables history of phi and theta.
- dump_mod: save history of phi and theta every n samples if view_mag_flag enabled.

Optional arguments (Used in the backend!):
- file_ID:      needed if parallelized as each concurrent sample must have a unique file ID.

Returns:
- bit sampled
- energy used in joules

Import this function in python with: 
`from interface_funcs import mtj_sample`

## Device class
Declare as one of:
`dev = SHE_MTJ_rng()`
`dev = VCMA_MTJ_rng()`
`dev = SWrite_MTJ_rng("<device flavor>")`

The SHE, and VCMA devices have default device parameters from a UT Austin fabbed device that can be set using `dev.set_vals()`.

The Stochastic Write device has two sets of parameters, one for a UT Austin device, and the other for a NYU device as described in 'Temperature-Resilient True Random Number Generation with Stochastic Actuated Magnetic Tunnel Junction Devices, Laura Rehm et. al. 2023'. This device needs to be declared with one of these options regardless of whether the parameters will be changed with the difference in this case being that the NYU device supports joule heating (see Joule Heating section). For example, `dev = SWrite_MTJ_rng("UTA")`

The default parameters are set in `mtj_parameters.json`


The devices have the following as modifiable parameters:
- T   [$`K`$]:  Temperature
- Ki  [$`\frac{J}{m^2}`$]:  Anisotropy energy
- Ms  [$`\frac{A}{m}`$] : Magnetic saturation
- tf  [$`m`$] : Thickness of the free layer
- tox [$`m`$] : Thickness of the oxide
- d   [$`m`$] : Thickness of the heavy metal layer
- a   [$`m`$]:  MTJ major ellipsoidal radius
- b   [$`m`$]:  MTJ minor ellipsoidal radius
- eta   [dimensionless] : Spin hall angle
- alpha [dimensionless] : Gilbert damping factor
- Rp   [$`\Omega`$] : Resistance in the parallel state
- TMR  [dimensionless] : Tunneling magneto resistance
- t_pulse  [$`s`$] : Pulse time
- t_relax  [$`s`$] : Relax time
- Nx  [dimensionless] : Demagnetization factors in x, y, and z
- Ny  [dimensionless]
- Nz  [dimensionless]

SHE only:
- J_she  [$`\frac{A}{m^2}`$] : SHE current density
- Hy (optional) : Applied magnetic field

VCMA only:
- v_pulse [$`V`$] : Voltage pulse

Stochastic Write only:

Anistropy and magnetic saturation are defined strictly at 295K. This enables the joule heating model for the NYU device. The current type implementation mandates that the UT Austin device follows the naming convention even without a joule heating model.

- K_295 [$`\frac{J}{m^2}`$] : Anisotropy at room temp
- Ms_295 [$`\frac{A}{m}`$] : Magnetic saturation at room temp
- J_reset [$`\frac{A}{m^2}`$] : Reset current density
- H_reset [$`\frac{A}{m}`$] : Reset applied magnetic field
- t_reset [$`s`$] : Reset time

## Device to Device / Cycle to Cycle variation
Device-to-device/cycle-to-cycle variation can be modeled crudely using a gaussian distrubition around a given device parameter using `vary_param(dev, param, std dev.)` in `mtj_helper.py`. This function takes a device, a named parameter, and the standard deviation for a gaussian distribution centered around the current device parameters value and returns a modified device. 

## Joule Heating
The Stochastic Write NYU device stack has a model of joule heating. This can be enabled/disabled with `dev.enable_heating()`/`dev.disable_heating()`. Currently no other device model is compatible with this model since the joule heating is dependent on the number of layers in the stack, materials, thicknesses, etc.


## Device Parameter Verification
`interface_funcs.py` also has a function, `mtj_check(device, J_stt, number_of_checks...)`, which will check the device parameters applied to the device to see:
1. Is the device physical?
2. Does the device go in-plane upon current application?
3. Does the device return to +- 1 when current is removed?

If yes to all three, then the configuration is good!

Import `from interface_funcs import mtj_check`,

and call `nerr, mz1, mz2, PI = mtj_check(dev, J_STT, number_of_checks)` 

For nerr, mz1, mz2, returned -1 is an error and 0 is a success. Positive integers are warnings.
for PI, 0 is success, -1 is PMA too strong, +1 is IMA too strong. Note that if there was numerical error, a -1 will be returned for nerr and `None` for everything else. 


## Slurm
Neither the fortran or python code is paralleized with openMP or MPI.

## Examples
```
# ===== test devices with mtj_check ======
from interface_funcs import mtj_sample, mtj_check
from mtj_helper import print_check
from mtj_types  import SHE_MTJ_rng, SWrite_MTJ_rng, VCMA_MTJ_rng

dev = SHE_MTJ_rng()
dev.init() # calls both set_vals and set_mag_vector with defaults
print(dev)

print_check(*mtj_check(dev, 0, 100))

dev = SWrite_MTJ_rng("UTA")
dev.init()
print(dev)

print_check(*mtj_check(dev, -1.2e11, 100))

dev = VCMA_MTJ_rng()
dev.init()
print(dev)

print_check(*mtj_check(dev, 0, 100))
```
```
# ===== generate scurve plots ======
from interface_funcs import mtj_sample
from mtj_types  import SWrite_MTJ_rng, SHE_MTJ_rng, VCMA_MTJ_rng
from mtj_helper import avg_weight_across_samples
import numpy as np
import matplotlib.pyplot as plt

class scurve:
  def __init__(self, dev, x, title):
    self.x = x
    self.y = self.generate(dev, num_to_avg = 1000)
    fig, ax = plt.subplots()
    ax.set_xlabel('J [A/m^2]')
    ax.set_title(title)
    self.ax = ax

  def generate(self, dev, num_to_avg):
      weights = []
      for J in self.x:
        weights.append(
          avg_weight_across_samples(dev, J, num_to_avg))
      return weights

j_steps = 50
SHE_current    = np.linspace(-6e9, 6e9, j_steps)
SWrite_current = np.linspace(-300e9, 0,  j_steps)
VCMA_current   = np.linspace(-6e9, 6e9, j_steps)

SWrite = SWrite_MTJ_rng("UTA")
SWrite.init()
print(SWrite)

SHE = SHE_MTJ_rng()
SHE.init()
print(SHE)

VCMA = VCMA_MTJ_rng()
VCMA.init()
print(VCMA)

curves = [ scurve( SHE, SHE_current, "SHE" ),
           scurve( SWrite, SWrite_current, "SWrite" ),
           scurve( VCMA, VCMA_current, "VCMA" )]

for c in curves: c.ax.plot(c.x, c.y), c.ax.scatter(c.x, c.y, s=36, marker='^')

plt.show()
```
