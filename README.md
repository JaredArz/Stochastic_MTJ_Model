# Stochastic MTJ device models using f2py
### jared.arzate@utexas.edu
##  Device models included:
### - [X] SOT  MTJ
### - [ ] VCMA MTJ
### - [ ] STT Stochastic Write MTJ

## Working :
### - [X] Serial
### - [ ] Parallel

## Fortran interface
navigate into the fortran_source directory and compile with:
```
cd ./fortran_source
make
```
The cpython binary this creates will compute the modified LLG code to pulse and relax an MTJ  

`interface_funcs.py` imports it as
```
import sys
sys.path.append("./fortran_source")
import single_sample as f90
```
and has a user function `mtj_sample(device, J_stt,...)` which will call this binary and handle the interface.


## Device class
Note that the updated device class only has the following available parameters:
- Ki
- Ms
- tf
- J she
- a
- b
- d
- eta
- alpha
- Rp
- TMR

The new device class also has updated usage with a flag to check if all parameters have been set.
```
# declaration takes in a dev-to-dev variation flag to assign noise if parameters have been made dev-to-dev dependent
dev = SHE_MTJ_rng(dd_flag = True)

# preset option available with set_vals(1), uses parameters manually defined in `mtj_types_v3.py`
dev.set_vals(1)

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
dev.tf=1
dev.J_she=1
dev.a=1
dev.b=1
dev.d=1
dev.eta=1
dev.alpha=1
dev.Rp=1
dev.TMR=1

/// parameter set flag is now true.

# or using set vals with as many parameters as needed
dev.set_vals(Ki=1,Ms=1)

/// (in no particular order)
dev.set_vals(Ki=1,Ms=1,tf=1,J_she=1,a=1,b=1,d=1,eta=1,alpha=1,Rp=1,TMR=1)

