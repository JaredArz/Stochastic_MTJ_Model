from interface_funcs import mtj_sample
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
def config_verify(dev):
    # Assuming no H field, v_pulse
    # J_she as set in device parameters
    cycles = 500
    steps = 1
    reps  = 1
    # This value should be the same as she_mtj_rng_params.f90
    t_step = 5e-11
    J_stt = np.linspace(0,0,steps)
    pulse_check_start = (1/5) * dev.t_pulse
    relax_check_start = (3/5) * dev.t_relax
    pulse_steps = int(np.ceil(dev.t_pulse/t_step))-1
    relax_steps = int(np.ceil(dev.t_relax/t_step))-1
    pulse_start = int(pulse_check_start/t_step)
    relax_start = int(relax_check_start/t_step)
    mz_avg = []
    mz_chk1_arr = np.zeros(pulse_steps)
    mz_chk2_arr = np.zeros(relax_steps)
    point2point_variation = 0
    for rep in range(reps):
        for id,j in enumerate(J_stt):
            #dev.set_mag_vector(0, np.pi/2)
            dev.set_mag_vector()
            mz_arr = []
            for i in range(cycles):
                # Only tracking magnetization vector stored in *History
                _,_ = mtj_sample(dev,J_stt,dump_mod=1,view_mag_flag=1,config_check=1)
                mz = np.cos(dev.thetaHistory)
                mz_arr.append(mz)
                mz_chk1_arr = mz_chk1_arr + np.absolute(mz[0:pulse_steps])
                mz_chk2_arr = mz_chk2_arr + np.absolute(mz[pulse_steps:pulse_steps+relax_steps])
                point2point_variation = point2point_variation + np.sum(np.absolute(mz[1::] - mz[0:-1]))/(mz.size - 1)
            mz_avg.append(np.mean(mz_arr))
    mz_chk1_arr = mz_chk1_arr/cycles
    mz_chk2_arr = mz_chk2_arr/cycles
    point2point_variation = point2point_variation/cycles

    mz_chk1_val = np.average(mz_chk1_arr[pulse_start:pulse_steps])
    mz_chk2_val = np.average(mz_chk2_arr[relax_start:relax_steps])

    mz_chk1_res = None
    mz_chk2_res = None
    PMAIMA      = None
    if point2point_variation > 0.25:
        numerical_err = -1
    else:
        numerical_err = 0

        if mz_chk1_val < 0.2:
            mz_chk1_res = 0
        elif mz_chk1_val < 0.5:
            mz_chk1_res = 1
        else:
            mz_chk1_res = -1

        if mz_chk2_val < 0.2:
            mz_chk2_res = -1
        elif mz_chk2_val < 0.5:
            mz_chk2_res = 1
        else:
            mz_chk2_res = 0

        if mz_chk1_res == -1:
            PMAIMA = -1 #if check 1 fails, then it is too PMA
        elif mz_chk2_res == -1:
            PMAIMA = 1 #if check 2 fails, it is not PMA enough
        else:
            PMAIMA = 0 #if both checks are just warnings or better, balance is good enough

    return numerical_err,mz_chk1_res,mz_chk2_res,PMAIMA



from mtj_types_v3 import SWrite_MTJ_rng
dev = SWrite_MTJ_rng()
dev.set_mag_vector()
dev.set_vals(0)
J_stt = 1.5e11
# NOTE: Ms found from Duc-The Ngo et al 2014 J. Phys. D: Appl. Phys. 47
# NOTE: Rp is RA/A assuming RA is Rp*A
# NOTE: Ki for now is using delta from the P->AP state (delta = 51). May need to add fortran code
#       to dectect state (and Temperature) and apply appropriate Ki value on the fly
dev.set_vals(a=40e-9, b=40e-9, TMR = 2.03, tf = 2.6e-9, Rp = 5530, Ki=0, Ms=0)
#dev.set_vals(a=40e-9, b=40e-9, TMR = 2.03, tf = 2.6e-9, Rp = 2530, Ki=0, Ms=0)
dev.set_vals(alpha=0.016)
_,_ = mtj_sample(dev,J_stt,dump_mod=1,view_mag_flag=1,config_check=1)
'''
nerr, mz1, mz2, PI = config_verify(dev)
# ignoring warnings
if nerr == -1:
    print('numerical error, do not use parameters!')
elif PI == -1:
    print('PMA too strong')
elif PI == 1:
    print('IMA too strong')
else:
    print('parameters okay')
    print("running application")
'''

cos = np.cos
sin = np.sin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pi = np.pi
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
plot = ax.plot_surface(sin(phi)*cos(theta),sin(phi)*sin(theta),cos(phi), rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.1)

x = sin(dev.thetaHistory)*cos(dev.phiHistory)
y = sin(dev.thetaHistory)*sin(dev.phiHistory)
z = cos(dev.thetaHistory)

t = range(0,217)
df = pd.DataFrame({"time" : t, "x" : x[0:217], "y" : y[0:217], "z" : z[0:217]})

def update_graph_scatter(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)

def update_graph_plot(num):
    data=df[df['time']==num]
    graph.set_data (data.x, data.y)
    graph.set_3d_properties(data.z)
    #title.set_text('3D Test, time={}'.format(num))
    #quiver.remove()
    #quiver = ax.quiver(*get_arrow(theta))
    #quiver = ax.quiver(0, 0, 0, data.x, data.y, data.z)
    return graph,# quiver

'''
def get_arrow(data):
    u = data.x
    v = data.y
    w = data.z
    return 0,0,0,u,v,w
'''

#quiver = ax.quiver(*get_arrow(df[df['time']==0]))

'''
def update(quiver, num):
    data=df[df['time']==num]
    quiver.remove()
    quiver = ax.quiver(*get_arrow(data))
    return quiver
'''

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("equal")

data=df[df['time']==0]
#graph = ax.scatter(data.x, data.y, data.z)
graph, = ax.plot(data.x, data.y, data.z, linestyle="solid", marker="o")
#global quiver
#quiver = ax.quiver(*get_arrow(data))
#ani = animation.FuncAnimation(fig, update_graph, 217, interval=1, blit=False)
ani = animation.FuncAnimation(fig, update_graph_plot, 217, interval=40, blit=True, repeat=True)
#ani2 = animation.FuncAnimation(fig, update, frames=217, fargs=(quiver,), interval=40, repeat=True)
plt.show()
