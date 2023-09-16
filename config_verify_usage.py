from config_verify import configuration_check
from mtj_types_v3 import SHE_MTJ_rng

# this is an example usage of checking a particular configuration for:
    # 1. Is the device physical?
    # 2. Does the device go in-plane upon current application?
    # 3. Does the device return to +- 1 when current is removed?
    # If yes to all three, then the configuration is good!
    # additional arguments (device parameters) can be added depending on what is changing
    # for nerr, mz1, mz2, returned -1 is an error and 0 is a success. positive integers are warnings
    # for PI, 0 is success, -1 is PMA too strong, +1 is IMA too strong

dev = SHE_MTJ_rng()
dev.set_vals(0)
nerr, mz1, mz2, PI = configuration_check(dev)
print(dev)

if nerr == -1:
    print('numerical error, do not use parameters!')
else:
    if   mz1 == 0:
        print('check1 success! Device goes in-plane with current application')
    elif mz1 == -1:
        print('check1 error, device does not go in-plane with current application')
    else:
        print('check1 warning; device partially goes in-plane with current application')

    if   mz2 == 0:
        print('check2 success! Device returns to +- 1 after current application')
    elif mz2 == -1:
        print('check2 error, device does not return to +- 1 after current application')
    else:
        print('check2 warning; device only partially returns to +- 1 after current application')

    if   PI == 0:
        print('total success! PMA/IMA balance is good')
    elif PI == -1:
        print('PMA too strong; try reducing ratio of Ki/Ms')
    else:
        print('IMA too strong; try increasing ratio of Ki/Ms')
