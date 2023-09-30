module sampling
    use MTJ_RNG_vars
    use ziggurat
    implicit none
    logical :: fwrite_enabled
    contains
        ! --------------------------------*------------*-----------------------------------
        !   energy_usage,bit,theta_end,phi_end, should not be passed into this function
        !   they are declared as they are because subroutine arguments declared with intent(out)
        !   will return back to python as a tuple (see numpy f2py)
        !
        !   intended to mimic the python function call: out,energy = dev.sample_*(Jappl,Jshe_in, self.theta, self.phi, self.Ki....)
        ! --------------------------------*------------*-----------------------------------
        subroutine sample_SHE(energy_usage, bit, theta_end, phi_end,&
                                 Jappl, Jshe, theta_init, phi_init, Ki_in, TMR_in, Rp_in,&
                                 a_in, b_in, tf_in, alpha_in, Ms_in, eta_in, d_in, t_pulse, t_relax,&
                                 dump_mod, view_mag_flag, sample_count, file_ID, config_check) 
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, Jshe, theta_init, phi_init, t_pulse, t_relax
            ! Device input parameters
            real, intent(in) :: Ki_in, TMR_in, Rp_in, Ms_in,&
                                a_in, b_in, d_in, tf_in, alpha_in, eta_in
            ! Functional parameters
            integer, intent(in) :: file_ID, sample_count, dump_mod, config_check
            logical, intent(in) :: view_mag_flag
            ! Return values
            real, intent(out) :: energy_usage, theta_end, phi_end
            integer, intent(out) :: bit
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, cuml_pow
            real(dp) :: phi_i, theta_i, Hk, V
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, arr_size
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init ========= 
            ! Fortran array indexing starts at 1
            t_i  = 1 
            fwrite_enabled = ((mod(sample_count,dump_mod) .eq. 0 .and. view_mag_flag) .or. config_check .eq. 1)

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)
            arr_size = pulse_steps+relax_steps+1

            allocate(cuml_pow(arr_size))
            cuml_pow(t_i) = 0.0_dp

            theta_i = real(theta_init, dp)
            phi_i   = real(phi_init, dp)
            if(fwrite_enabled) then
                allocate(theta_evol(arr_size))
                allocate(phi_evol(arr_size))
                theta_evol(t_i) = theta_i
                phi_evol(t_i)   = phi_i
            end if

            call set_params(Ki_in, TMR_in, Rp_in, Ms_in, alpha_in, tf_in, a_in, b_in, d_in, eta_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            !=========== Pulse current and set device to be in-plane =========
            V = 0.0_dp
            Hk = (2.0_dp*Ki)/(tf*Ms*u0)-(2.0_dp*ksi*V)/(u0*tox*tf)
            call drive(V, real(Jshe,dp), real(Jappl,dp), Hk, pulse_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, cuml_pow)

            !=================  Relax into one of two low-energy states out-of-plane  ===================
            V = 0.0_dp
            Hk = (2.0_dp*Ki)/(tf*Ms*u0)-(2.0_dp*ksi*V)/(u0*Ms*tox*tf)
            call drive(V, 0.0_dp, real(Jappl,dp), Hk, relax_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, cuml_pow)

            if(fwrite_enabled) then
                call file_dump(file_ID, phi_evol, theta_evol)
            end if

            ! ===== return final solve values: energy,bit,theta,phi ====
            if( cos(theta_i) > 0.0_dp ) then
                bit = 1
            else
                bit = 0
            end if
            ! Cast to real before returning to Python
            theta_end = real(theta_i)
            phi_end   = real(phi_i)
            energy_usage = real(sum(cuml_pow)*t_step)
        end subroutine sample_SHE

        subroutine sample_VCMA(energy_usage, bit, theta_end, phi_end,&
                                 Jappl, v_pulse, theta_init, phi_init, Ki_in, TMR_in, Rp_in,&
                                 a_in, b_in, tf_in, alpha_in, Ms_in, eta_in, d_in, t_pulse, t_relax,&
                                 dump_mod, view_mag_flag, sample_count, file_ID, config_check) 
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, v_pulse, theta_init, phi_init, t_pulse, t_relax
            ! Device input parameters
            real, intent(in) :: Ki_in, TMR_in, Rp_in, Ms_in,&
                                a_in, b_in, d_in, tf_in, alpha_in, eta_in
            ! Functional parameters
            integer, intent(in) :: file_ID, sample_count, dump_mod, config_check
            logical, intent(in) :: view_mag_flag
            ! Return values
            real, intent(out) :: energy_usage, theta_end, phi_end
            integer, intent(out) :: bit
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, cuml_pow
            real(dp) :: phi_i, theta_i, Hk, V
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, arr_size
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init ========= 
            ! Fortran array indexing starts at 1
            t_i  = 1 
            fwrite_enabled = ((mod(sample_count,dump_mod) .eq. 0 .and. view_mag_flag) .or. config_check .eq. 1)

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)
            arr_size = pulse_steps+relax_steps+1

            allocate(cuml_pow(arr_size))
            cuml_pow(t_i) = 0.0_dp

            theta_i = real(theta_init, dp)
            phi_i   = real(phi_init, dp)
            if(fwrite_enabled) then
                allocate(theta_evol(arr_size))
                allocate(phi_evol(arr_size))
                theta_evol(t_i) = theta_i
                phi_evol(t_i)   = phi_i
            end if

            call set_params(Ki_in, TMR_in, Rp_in, Ms_in, alpha_in, tf_in, a_in, b_in, d_in, eta_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            V = real(v_pulse,dp)
            Hk = (2.0_dp*Ki)/(tf*Ms*u0)-(2.0_dp*ksi*V)/(u0*tox*tf)
            call drive(V, 0.0_dp, real(Jappl,dp), Hk, pulse_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, cuml_pow)

            V = 0.0_dp
            Hk = (2.0_dp*Ki)/(tf*Ms*u0)-(2.0_dp*ksi*V)/(u0*Ms*tox*tf)
            call drive(V, 0.0_dp, real(Jappl,dp), Hk, relax_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, cuml_pow)

            if(fwrite_enabled) then
                call file_dump(file_ID, phi_evol, theta_evol)
            end if

            ! ===== return final solve values: energy,bit,theta,phi ====
            if( cos(theta_i) > 0.0_dp ) then
                bit = 1
            else
                bit = 0
            end if
            ! Cast to real before returning to Python
            theta_end = real(theta_i)
            phi_end   = real(phi_i)
            energy_usage = real(sum(cuml_pow)*t_step)
        end subroutine sample_VCMA

        subroutine drive(V, J_SHE, J_STT, Hk, steps, t_i, phi_i, theta_i, phi_evol, theta_evol, cuml_pow)
           implicit none
           integer,  parameter :: dp = kind(0.0d0)
           real(dp), intent(in) :: V, J_SHE, J_STT, Hk
           integer,  intent(in)  :: steps
           real(dp), dimension(:), intent(inout) :: phi_evol, theta_evol, cuml_pow
           real(dp), intent(inout) :: phi_i, theta_i
           integer,  intent(inout) :: t_i
           real(dp) :: Ax, Ay, Az, dphi, dtheta, R1, pow
           integer  :: i

           do i = 1, steps
               t_i = t_i+1
               Ax = Hx-Nx*Ms*sin(theta_i)*cos(phi_i)     +rnor()*Htherm
               Ay = Hy-Ny*Ms*sin(theta_i)*sin(phi_i)     +rnor()*Htherm
               Az = Hz-Nz*Ms*cos(theta_i)+Hk*cos(theta_i)+rnor()*Htherm

               dphi   = gammap*(Ax*(-cos(theta_i)*cos(phi_i)-alpha*sin(phi_i))+Ay*(-cos(theta_i)*sin(phi_i)+alpha*cos(phi_i))+&
                   Az*sin(theta_i))/(sin(theta_i))+J_SHE*F*eta*(sin(phi_i)-alpha*cos(phi_i)*cos(theta_i))/(sin(theta_i)*&
                   (1_dp+alpha*alpha))-((alpha*F*P*J_STT)/(1_dp+alpha**2))
               dtheta = gammap*(Ax*(alpha*cos(theta_i)*cos(phi_i)-sin(phi_i))+Ay*(alpha*cos(theta_i)*sin(phi_i)+cos(phi_i))-Az*&
                   alpha*sin(theta_i))-J_SHE*F*eta*(cos(phi_i)*cos(theta_i)+(alpha*sin(phi_i))/(1_dp+alpha**2))+((F*P*J_STT)*&
                   sin(theta_i)/(1_dp+alpha**2))
               R1 = Rp*(1_dp+(V/Vh)**2+TMR)/(1_dp+(V/Vh)**2+TMR*(1_dp+(cos(theta_i)))/2_dp)

               pow = 0.5_dp*cap_mgo*V**2+R2*(J_SHE*A2)**2+R1*(J_STT*A1)**2
               phi_i   = phi_i+t_step*dphi 
               theta_i = theta_i+t_step*dtheta
               cuml_pow(t_i) = pow
               if(fwrite_enabled) then
                   theta_evol(t_i) = theta_i
                   phi_evol(t_i)   = phi_i
               end if
           end do
        end subroutine drive

        subroutine file_dump(file_ID, phi_evol, theta_evol)
            implicit none
            integer,  parameter :: dp = kind(0.0d0)
            real(dp), dimension(:), intent(in) :: phi_evol, theta_evol
            integer,  intent(in) :: file_ID
            character(len=7) :: file_string

            ! ===== array dump to file of theta/phi time evolution  ====
            write (file_string,'(I7.7)') file_ID
            open(unit = file_ID, file = "time_evol_mag_"//file_string//".txt", action = "write", status = "replace", &
                    form = 'formatted')
            write(file_ID,*) phi_evol
            write(file_ID,*) theta_evol
            close(file_ID)
        end subroutine file_dump
end module sampling
