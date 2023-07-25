! -------------------------*---------*--------------------------
! These moudles contain the experimental device parameters for 
! a particular device corresponding to each module name.
! All values below are asummed to be fixed and any device-to-device
! or cycle-to-cycle variation is stored within the python dev class.
! -------------------------*---------*--------------------------

!these variables do not change
module SHE_MTJ_rng_params
    implicit none
    integer,parameter :: dp = kind(0.0d0)
    real(dp),parameter :: pi    = real(4.0,dp)*DATAN(real(1.0,dp))
    real(dp),parameter :: uB    = 9.274e-24
    real(dp),parameter :: h_bar = 1.054e-34          
    real(dp),parameter :: u0    = pi*4e-7               
    real(dp),parameter :: e     = 1.6e-19                
    real(dp),parameter :: kb    = 1.38e-23   
    real(dp),parameter :: gammall = 2.0*u0*uB/h_bar 
    real(dp),parameter :: gammab  = gammall/u0
    real(dp),parameter :: t_step = 5e-11
    real(dp),parameter :: t_pulse = 10e-9
    real(dp),parameter :: v_pulse = 0.0
    real(dp),parameter :: t_relax = 15e-9
    real(dp),parameter :: tox = 1.5e-9
    real(dp),parameter :: P   = 0.6
    real(dp),parameter :: T     = 300.0
    real(dp),parameter :: ksi = 75e-15
    real(dp),parameter :: Vh    = 0.5
    real(dp),parameter :: w = 100e-9
    real(dp),parameter :: l = 100e-9
    real(dp),parameter :: RA = 7e-12
    real(dp),parameter :: rho = 200e-8
    real(dp),parameter :: eps_mgo = 4.0
    real(dp),parameter :: Hx = 0.0
    real(dp),parameter :: Hy = 0.0
    real(dp),parameter :: Hz = 0.0
    real(dp),parameter :: Nx = 0.010613177892974
    real(dp),parameter :: Ny = 0.010613177892974
    real(dp),parameter :: Nz = 0.978773644214052
    integer,parameter :: relax_steps = int(t_relax/t_step)
    integer,parameter :: pulse_steps = int(t_pulse/t_step)
    ! ======  used to calculate device Rp ==========
    ! ==== Calculation not used in current model ===
    ! real(dp),parameter :: delta = 40.0 
    ! real(dp),parameter :: Eb  = delta*kb*T
    ! ==============================================
end module SHE_MTJ_rng_params

!FIXME: add module for VCMA driven mtj device
