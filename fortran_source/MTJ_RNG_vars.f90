! --------------------------------*---------*----------------------------------------
! This module contains some experimental device parameters for an 
! MTJ random number generator device as well as placeholders for variable parameters
! --------------------------------*---------*----------------------------------------

module MTJ_RNG_vars
    implicit none
    ! Constants which should not be changed in current model.
    real(kind(0.0d0)),parameter :: pi    = real(4.0,kind(0.0d0))*DATAN(real(1.0,kind(0.0d0)))
    real(kind(0.0d0)),parameter :: uB    = 9.2740100783e-24
    real(kind(0.0d0)),parameter :: h_bar = 1.05457182e-34         
    real(kind(0.0d0)),parameter :: u0    = pi*(4e-7)               
    real(kind(0.0d0)),parameter :: e     = 1.602176634e-19
    real(kind(0.0d0)),parameter :: kb    = 1.380649e-23
    real(kind(0.0d0)),parameter :: eps_not = 8.8541878128e-12
    real(kind(0.0d0)),parameter :: gammall = 2.0*u0*uB/h_bar 
    real(kind(0.0d0)),parameter :: gammab  = gammall/u0
    real(kind(0.0d0)),parameter :: t_step  = 5e-11
    real(kind(0.0d0)),parameter :: P = 0.6
    real(kind(0.0d0)),parameter :: eps_mgo = 4.0

    real(kind(0.0d0)),parameter :: ksi = 75e-15
    real(kind(0.0d0)),parameter :: Vh  = 0.5
    real(kind(0.0d0)),parameter :: w = 100e-9
    real(kind(0.0d0)),parameter :: l = 100e-9
    real(kind(0.0d0)),parameter :: rho = 200e-8
    ! NOTE: Nx,Ny,Nz are experimental values dependent on device geometry. 
    real(kind(0.0d0)),parameter :: Nx = 0.138391020053
    real(kind(0.0d0)),parameter :: Ny = 0.138391020053
    real(kind(0.0d0)),parameter :: Nz = 0.723217959894
    real(kind(0.0d0)) :: Hx = 0.0
    real(kind(0.0d0)) :: Hy = 0.0
    real(kind(0.0d0)) :: Hz = 0.0
    real(kind(0.0d0)) :: Ki, Ms, T,&
                         Bsat, TMR, Rp, a, b, d, tf, alpha, eta, tox,&
                         gammap, volume, A1, A2, cap_mgo, R2, Htherm, F

    ! Modeling the stack for thermal module
    integer :: nLayers = 8
    real(kind(0.0d0)) :: k_j(2)
    real(kind(0.0d0)), allocatable :: tLayer(:), pLayer(:), cLayer(:)

    contains
        !NOTE: requires Ms, K, and T to be set
        subroutine set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in,&
                              d_in, eta_in, tox_in)
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            real, intent(in) :: TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in,&
                                d_in, eta_in, tox_in

            if (Ki .eq. 0.0_dp .or. Ms .eq. 0.0_dp) then
                print *, "=============== FORTRAN RUNTIME ERROR ==================="
                print *, "= parameters not set properly before calling set_params ="
                print *, "========================================================="
                print *, "========================================================="
                stop
            end if
            TMR = real(TMR_in, dp); Rp = real(Rp_in, dp)
            a  = real(a_in, dp);   b   = real(b_in, dp)
            d  = real(d_in, dp);   eta = real(eta_in, dp)
            alpha = real(alpha_in, dp); tf = real(tf_in, dp)
            tox = real(tox_in,dp)

            A1      = a*b*pi/4.0_dp
            Bsat    = Ms*u0
            gammap  = gammall/(1.0_dp+alpha*alpha)
            volume  = tf*pi*b*a/4.0_dp
            cap_mgo = eps_not*eps_mgo*A1/tox 
            Htherm  = sqrt((2.0_dp*u0*alpha*kb*T)/(Bsat*gammab*t_step*volume))/u0
            F       = (gammall*h_bar)/(2.0_dp*u0*e*tf*Ms)
            A2      = d*w
            R2      = rho*l/(w*d)
        end subroutine set_params
end module MTJ_RNG_vars
