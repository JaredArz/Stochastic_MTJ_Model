module single_sample
    use she_mtj_rng_params
    use ziggurat
    implicit none
    integer             :: i,t_iter
    ! dummy variables
    integer :: pulse_steps,relax_steps
    real,allocatable,target   :: theta_over_time(:),phi_over_time(:),cumulative_pow(:)
    real,dimension(:),pointer :: th_ptr, phi_ptr, pow_ptr

    real     :: Jappl,Jshe,theta_init,phi_init,t_pulse,t_relax
    ! Device parameters
    real     :: Ki,TMR,Rp,a,b,tf,alpha,Ms,eta,d

    logical          :: w_enabled
    real(dp) :: V,J_SHE,J_STT,Hk,Ax,Ay,Az,dphi,dtheta,R1
    real(dp) :: phi_i,theta_i,power_i,seed
    real(dp) :: Bsat,gammap,volume,A1,A2,cap_mgo,R2,Htherm,F
    contains
        ! --------------------------------*------------*-----------------------------------
        !   energy_usage,bit,theta_end,phi_end, should not be passed into this function
        !   they are declared as they are because subroutine arguments declared with intent(out)
        !   will return back to python as a tuple (not sure exactly why, see numpy f2py)
        !
        !
        !   intended to mimic the python function call: out,energy = dev.sample(Jappl,Jshe_in, self.theta, self.phi, self.Ki....)
        !
        ! --------------------------------*------------*-----------------------------------
        subroutine sample(energy_usage,bit,theta_end,phi_end,&
                                    Jappl,Jshe,theta_init,phi_init,Ki,TMR,&
                                    Rp,a,b,tf,alpha,Ms,eta,d,t_pulse,t_relax,mtj_type,&
                                    dump_mod,view_mag_flag,sample_count,file_ID,config_check) 
        implicit none
        ! Dynamical parameters
        real,intent(in)     :: Jappl_i,Jshe_i,theta_init_i,phi_init_i,t_pulse_i,t_relax_i
        ! Device parameters
        real,intent(in)     :: Ki_i,TMR_i,Rp_i,a_i,b_i,tf_i,alpha_i,Ms_i,eta_i,d_i
        ! =================
        integer, intent(in) :: file_ID, sample_count,dump_mod,config_check,mtj_type
        logical, intent(in) :: view_mag_flag
        !return values
        real,intent(out) :: energy_usage,theta_end,phi_end
        integer,intent(out) :: bit
        !==================================================================
        !//////////////////////////////////////////////////////////////////
        Jappl=Jappl_i
        Jshe=Jshe_i
        theta_init=theta_init_i
        phi_init=phi_init_i
        t_pulse=t_pulse_i
        t_relax=t_relax_i
        Ki=Ki_i
        TMR=TMR_i
        Rp=Rp_i
        a=a_i
        b=b_i
        tf=tf_i
        alpha=alpha_i
        Ms=Ms_i
        eta=eta_i
        d=d_i

        w_enabled = ((mod(sample_count,dump_mod) .eq. 0 .and. view_mag_flag) .or. config_check .eq. 1)
        ! ==== parameters with variable dependencies ====
        pulse_steps = int(t_pulse/t_step)
        relax_steps = int(t_relax/t_step)
        Bsat    = real(Ms,dp)*u0
        gammap  = gammall/(1.0_dp+real(alpha,dp)*real(alpha,dp))
        volume  = real(tf,dp)*pi*real(b,dp)*real(a,dp)/4.0_dp
        A1      = real(a,dp)*real(b,dp)*pi/4.0_dp
        A2      = real(d,dp)*w
        cap_mgo = 8.854e-12_dp*eps_mgo*A1/tox 
        R2      = rho*l/(w*real(d,dp))
        Htherm  = sqrt((2.0_dp*u0*real(alpha,dp)*kb*T)/(Bsat*gammab*t_step*volume))/u0
        F       = (gammall*h_bar)/(2.0_dp*u0*e*real(tf,dp)*real(Ms,dp))
        ! =================================================================

        !  static variables do not persist back in python so zigset (rng init function) is called each time this code runs
        !  This code is a module for an existing object-oriented code so the return values from this function               
        !  should be made to update any existing objects if necessary                                                      
        call random_number(seed)
        call zigset(int(1+floor((1000001)*seed)))

        ! ======== solve init ========= 
        t_iter  = 1 ! fortran has array indexing of 1, in math terms, t=0
        power_i = 0.0
        theta_i = real(theta_init,dp)
        phi_i   = real(phi_init,dp)
        if( w_enabled ) then
            theta_over_time(t_iter) = real(theta_i)
            phi_over_time(t_iter)   = real(phi_i)
        end if
        cumulative_pow(t_iter) = real(power_i)

        if( mtj_type .eq. 0 ) then
            call SHE()
        else if( mtj_type .eq. 1 ) then
            call SWrite()
        else if( mtj_type .eq. 2 ) then
            call VCMA()
        end if

        ! ===== array dump to file of theta/phi time evolution  ====
        if( w_enabled ) then
            call file_dump(file_ID)
        end if
        ! ========================================================== 

        ! ===== return final solve values: energy,bit,theta,phi ====
        theta_end = real(theta_i)
        phi_end   = real(phi_i)
        if( cos(theta_end) > 0.0 ) then
            bit = 1
        else
            bit = 0
        end if
        energy_usage = real(sum(cumulative_pow))*real(t_step)
        contains
            subroutine file_dump(file_ID) !use pointers for the two arrays
                implicit none
                integer,intent(IN) :: file_ID
                character(len=7)   :: file_string
                write (file_string,'(I7.7)') file_ID
                open(unit = file_ID, file = "time_evol_mag_"//file_string//".txt", action = "write", status = "replace", &
                        form = 'formatted')
                write(file_ID,*) phi_over_time
                write(file_ID,*) theta_over_time
                close(file_ID)
            end subroutine file_dump
        end subroutine sample

        subroutine SHE()
            if( w_enabled ) then
                allocate(theta_over_time(pulse_steps+relax_steps+1))
                allocate(phi_over_time(pulse_steps+relax_steps+1))
            end if
            allocate(cumulative_pow(pulse_steps+relax_steps+1))

            !====================== Pulse current and set device to be in-place ======================
            V     = 0
            J_SHE = real(Jshe,dp)
            J_STT = real(Jappl,dp)
            Hk    = (2.0*real(Ki,dp))/(real(tf,dp)*real(Ms,dp)*u0)-(2.0_dp*ksi*V)/(u0*tox*real(tf,dp))
            call pulse(V,J_SHE,J_STT,Hk)
            !=================  Relax into a one of two low-energy states out-of-plane  ===================

            V=0
            J_SHE = 0.0
            J_STT = real(Jappl,dp)
            Hk = (2.0*real(Ki,dp))/(real(tf,dp)*real(Ms,dp)*u0)-(2.0_dp*ksi*V)/(u0*real(Ms,dp)*tox*real(tf,dp))
            call relax(V,J_SHE,J_STT,Hk)
        end subroutine

        subroutine SWrite()
            if( w_enabled ) then
                allocate(theta_over_time(pulse_steps+relax_steps+1))
                allocate(phi_over_time(pulse_steps+relax_steps+1))
            end if
            allocate(cumulative_pow(pulse_steps+relax_steps+1))
        end subroutine

        subroutine VCMA()
            implicit none
            if( w_enabled ) then
                allocate(theta_over_time(pulse_steps+relax_steps+1))
                allocate(phi_over_time(pulse_steps+relax_steps+1))
            end if
            allocate(cumulative_pow(pulse_steps+relax_steps+1))
        end subroutine


        subroutine pulse(V,J_SHE,J_STT,Hk)
            implicit none
            real(dp),intent(IN) :: V,J_SHE,J_STT,Hk
            do i = 1, pulse_steps
                !keep track of time steps for array navigation
                t_iter=i+1
                Ax = Hx-Nx*Ms*sin(theta_i)*cos(phi_i)     +rnor()*Htherm
                Ay = Hy-Ny*Ms*sin(theta_i)*sin(phi_i)     +rnor()*Htherm
                Az = Hz-Nz*Ms*cos(theta_i)+Hk*cos(theta_i)+rnor()*Htherm

                dphi   = gammap*(Ax*(-cos(theta_i)*cos(phi_i)-alpha*sin(phi_i))+Ay*(-cos(theta_i)*sin(phi_i)+alpha*cos(phi_i))+&
                    Az*sin(theta_i))/(sin(theta_i))+J_SHE*F*eta*(sin(phi_i)-alpha*cos(phi_i)*cos(theta_i))/(sin(theta_i)*&
                    (1+alpha*alpha))-((alpha*F*P*J_STT)/(1+alpha**2))
                dtheta = gammap*(Ax*(alpha*cos(theta_i)*cos(phi_i)-sin(phi_i))+Ay*(alpha*cos(theta_i)*sin(phi_i)+cos(phi_i))-Az*&
                    alpha*sin(theta_i))-J_SHE*F*eta*(cos(phi_i)*cos(theta_i)+(alpha*sin(phi_i))/(1+alpha**2))+((F*P*J_STT)*&
                    sin(theta_i)/(1+alpha**2))
                R1     = real(Rp,dp)*(1+(V/Vh)**2+real(TMR,dp))/(1+(V/Vh)**2+real(TMR,dp)*(1+(cos(theta_i)))/2)
                power_i = 0.5*cap_mgo*V**2+R2*(abs(J_SHE*A2))**2+R1*(J_STT*A1)**2
                phi_i   = phi_i+t_step*dphi 
                theta_i = theta_i+t_step*dtheta
                cumulative_pow(t_iter) = real(power_i)
                if( w_enabled ) then
                    theta_over_time(t_iter) = real(theta_i)
                    phi_over_time(t_iter)   = real(phi_i)
                end if
            end do
        end subroutine
        subroutine relax(V,J_SHE,J_STT,Hk)
            implicit none
            real(dp),intent(IN) :: V,J_SHE,J_STT,Hk
            do i = 1, relax_steps
                t_iter=t_iter+1
                Ax = Hx-Nx*Ms*sin(theta_i)*cos(phi_i)     +rnor()*Htherm
                Ay = Hy-Ny*Ms*sin(theta_i)*sin(phi_i)     +rnor()*Htherm
                Az = Hz-Nz*Ms*cos(theta_i)+Hk*cos(theta_i)+rnor()*Htherm

                dphi = gammap*(Ax*(-cos(theta_i)*cos(phi_i)-alpha*sin(phi_i))+Ay*(-cos(theta_i)*sin(phi_i)+alpha*cos(phi_i))+&
                    Az*sin(theta_i))/(sin(theta_i))+J_SHE*F*eta*(sin(phi_i)-alpha*cos(phi_i)*cos(theta_i))/(sin(theta_i)*&
                    (1+alpha**2))-((alpha*F*P*J_STT)/(1+alpha**2))
                dtheta = gammap*(Ax*(alpha*cos(theta_i)*cos(phi_i)-sin(phi_i))+Ay*(alpha*cos(theta_i)*sin(phi_i)+cos(phi_i))-&
                    Az*alpha*sin(theta_i))-J_SHE*F*eta*(cos(phi_i)*cos(theta_i)+(alpha*sin(phi_i))/(1+alpha**2))+((F*P*J_STT)*&
                    sin(theta_i)/(1+alpha**2))

                R1 = real(Rp,dp)*(1+(V/Vh)**2+real(TMR,dp))/(1+(V/Vh)**2+real(TMR,dp)*(1+(cos(theta_i)))/2)
                power_i = 0.5*cap_mgo*V**2+R2*(abs(J_SHE*A2))**2+R1*(J_STT*A1)**2
                phi_i   = phi_i+t_step*dphi
                theta_i = theta_i+t_step*dtheta
                cumulative_pow(t_iter) = real(power_i)
                if( w_enabled ) then
                    theta_over_time(t_iter) = real(theta_i)
                    phi_over_time(t_iter)   = real(phi_i)
                end if
            end do
        end subroutine
        subroutine reset()
            implicit none

        end subroutine
end module single_sample
