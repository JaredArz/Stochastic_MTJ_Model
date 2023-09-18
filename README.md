# Stochastic MTJ device models
### contact: jared.arzate@utexas.edu
##  Device models included:
### - [X] SOT  MTJ
### - [ ] VCMA MTJ
### - [ ] STT Stochastic Write MTJ

## Working:
### - [X] Serial
### - [ ] Parallel

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
a SHE_MTJ_rng device (class described below) passed in as the first argument.
This device class should be intialized with device parameters and an initial magnetization vector.

The other arguments are as follows:
- Jstt:     spin-transfer torque current to apply.
- dump_mod: save history of phi and theta every n samples if view_mag_flag enabled.
- view_mag_flag: enables/disables history of phi and theta.


Optional arguments (Used in the backend!):
- file_ID:      needed if parallelized as each concurrent sample must have a unique file ID.
- config_check: to bet set if the device parameters are being verified. 

Returns:
- bit sampled
- energy used

Import `mtj_sample(...)` in python with `from interface_funcs import mtj_sample`

## Device class
Declare as `dev = SHE_MTJ_rng()`

Set default device parameters with `dev.set_vals(0)`,

Or add device-to-device variation with a 5% normally distributed TMR, Rp, Ki following default parameters by using `dev.set_vals(1)`

If setting the device parameters manually, the following must be set:
- Ki [J/m^2]
- Ms [A/m]
- tf [m]
- J_she  [A/m^2]
- a  [m]
- b  [m]
- d  [m]
- eta   [dimensionless]
- alpha [dimensionless]
- Rp   [$\`Ohm$\`]
- TMR  [dimensionless]  
- t_pulse  [t]
- t_relax  [t]


## Device Parameter Verification
`config_verify.py` is a code to check:
1. Is the device physical?
2. Does the device go in-plane upon current application?
3. Does the device return to +- 1 when current is removed?

If yes to all three, then the configuration is good!

Import `from config_verify import config_verify`,

and call `nerr, mz1, mz2, PI = config_verify(my_dev)` 

For nerr, mz1, mz2, returned -1 is an error and 0 is a success. Positive integers are warnings.
for PI, 0 is success, -1 is PMA too strong, +1 is IMA too strong

## Scripts
`mtj_dist_gen.py` is an example script using the MTJ to random numbers from exponential distribution.

`mtj_param_sweep.py` is an example script sweeping parameters.


Note that the above two scripts are no longer maintained.

## Slurm
Neither the fortran or python code is paralleized with openMP or MPI.

## Examples 
```
Misc.

#set (phi,theta)
dev.set_mag_vector(3.14/2, 3.14/2)

/// or

dev.phi = 3.14/2
dev.theta = 3.14/2

# print device parameters
print(dev)

# Assign individually
dev.Ki=1
dev.Ms=1
...

# or using set vals with as many parameters as needed
dev.set_vals(Ki=1,Ms=1)

/// (in no particular order)
dev.set_vals(Ki=1,Ms=1,tf=1,J_she=1,a=1,b=1,d=1,eta=1,alpha=1,Rp=1,TMR=1)

