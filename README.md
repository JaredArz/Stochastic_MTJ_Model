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
- T:        Temperature in kelvin
- dump_mod: save history of phi and theta every n samples if view_mag_flag enabled.
- view_mag_flag: enables/disables history of phi and theta.


Optional arguments (Used in the backend!):
- file_ID:      needed if parallelized as each concurrent sample must have a unique file ID.

Returns:
- bit sampled
- energy used

Import this function in python with: 
`from interface_funcs import mtj_sample`

## Device class
Declare as one of:
`dev = SHE_MTJ_rng()`
`dev = VCMA_MTJ_rng()`
`dev = SWrite_MTJ_rng()`

The SHE, and VCMA devices have default device parameters from a UT Austin fabbed device that can be set using `dev.set_vals()`. The Stochastic Write device has two sets of parameters, one for a UT Austin device, and the other for a NYU device as described in 'Temperature-Resilient True Random Number Generation with Stochastic Actuated Magnetic Tunnel Junction Devices, Laura Rehm et. al. 2023'

Device-to-device/cycle-to-cycle variation can be modeled using a simple gaussian distrubition around a given device parameter using `vary_param(dev, param, std dev.)` in `mtj_helper.py` 

If setting all the device parameters manually, the following must be set:
- Ki  [$`\frac{J}{m^2}`$]
- Ms  [$`\frac{A}{m}`$]
- tf  [$`m`$]
- tox [$`m`$]
- a   [$`m`$]
- b   [$`m`$]
- d   [$`m`$]
- eta   [dimensionless]
- alpha [dimensionless]
- Rp   [$`\Omega`$]
- TMR  [dimensionless]
- t_pulse  [$`t`$]
- t_relax  [$`t`$]

SHE only:
- J_she  [$`\frac{A}{m^2}`$]
- Hy (optional)

VCMA only:
- v_pulse [volts]

Stochastic Write only:
Anistropy and magnetic saturation are only defined at 295K (temperature dependence):
- K_295
- Ms_295
- J_reset
- H_reset
- t_reset

## Joule Heating
The Stochastic Write NYU device stack has a model of joule heating. This can be enabled with `dev.enable_heating()`. Currently no other device model is compatible with this model since the joule heating is dependent on the number of layers in the stack, materials, thicknesses, etc.


## Device Parameter Verification
`interface_funcs.py` also has a function for the SHE MTJ, `mtj_check(device, J_stt, number_of_checks...)`, which will check the device parameters applied to the device to see:
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
Misc.

#set (phi,theta)
dev.set_mag_vector(3.14/2, 3.14/2)

# print device parameters
print(dev)

# set default device parameters with
dev.set_vals()

# or use set vals to specify some parameters
dev.set_vals(Ki=1,Ms=1)

