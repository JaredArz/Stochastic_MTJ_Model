import numpy as np

def draw_norm(x,psig):
    return (x*np.random.normal(1,psig))

def draw_const(x,csig):
    return (x+np.random.normal(-csig,csig))

# Device-to-device variation and cycle-to-cycle variation
# can be modeled simply with this function.
# Takes a device, device parameter and a percent deviation
# which defines the standard deviation of a sampled
# gaussian distribution around the current parameter value.
# Returns the modified device.
def vary_param(dev, param, stddev):
    current_val = dev.__getattribute__(param)
    updated_val = draw_norm(current_val, stddev)
    dev.__setattr__(param,updated_val)
    return dev
