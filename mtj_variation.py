
#============================== Private ========================================
# inject noise if device-to-device variation requested
def draw_norm(x,var,psig):
    return (x if not var else(x*np.random.normal(1,psig)))

def draw_const(x,var,csig):
    return (x if not var else(x+np.random.normal(-csig,csig)))
