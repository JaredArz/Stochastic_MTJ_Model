! ================================================================================================
! pseudo RNG from: Marsaglia, G. & Tsang, W.W. (2000) `The ziggurat method for generating
! random variables', J. Statist. Software, v5(8).
! translated from C by Alan Miller (amiller@bigpond.net.au)
! Edits of this code and an implementatoin of the Box-Muller method for generating random numbers
! have been made by Jared Arzate (jared.arzate@utexas.edu)
! =================================================================================================
module ziggurat
   implicit none
   private

   integer,parameter:: dpz = selected_real_kind(12,60)
   logical,  save            ::  initialized=.FALSE.
   real(dpz), parameter  ::  m1=2147483648.0_dpz,   m2=2147483648.0_dpz, half=0.5_dpz
   real(dpz)             ::  dn, tn, vn, de, te, ve, q
   integer,  save            ::  iz, jz, jsr, hz
   ! ====================================================================
   ! ========= avoiding allocatable arrays for pythons sake =============
   ! ==== all corresponding code is commented for this reason  ========== 
   ! ====================================================================
   !integer, allocatable, save        :: kn(:), ke(:)
   !real, allocatable, save  :: wn(:), fn(:), we(:), fe(:)
   integer, save        :: kn(0:127), ke(0:255)
   real(dpz), save  :: wn(0:127), fn(0:127), we(0:255), fe(0:255)

   PUBLIC  :: zigset, shr3, uni, rnor!, cleanZig

contains
    subroutine zigset( jsrseed )
       integer, intent(in)  :: jsrseed
       integer  :: i

       !  Set the seed
       jsr = jsrseed

       !allocate(wn(128))
       !allocate(fn(128))
       !allocate(we(256))
       !allocate(fe(256))

       !allocate(kn(128))
       !allocate(ke(256))

       dn=3.442619855899_dpz
       tn=3.442619855899_dpz
       vn=0.00991256303526217_dpz
       de=7.697117470131487_dpz
       te=7.697117470131487_dpz
       ve=0.003949659822581572_dpz

       !  Tables for RNOR
       q = vn*exp(half*dn*dn)
       kn(0) = int((dn/q)*m1)
       kn(1) = 0
       wn(0) = q/m1
       wn(127) = dn/m1
       fn(0) = 1.0_dpz
       fn(127) = exp( -half*dn*dn )
       do  i = 126, 1, -1
          dn = SQRT( -2.0_dpz * LOG( vn/dn + exp( -half*dn*dn ) ) )
          kn(i+1) = int((dn/tn)*m1)
          tn = dn
          fn(i) = exp(-half*dn*dn)
          wn(i) = dn/m1
       end do

       initialized = .TRUE.
       RETURN
    end subroutine zigset

    !subroutine cleanZig()
    !   implicit none
    !   deallocate(wn)
    !   deallocate(fn)
    !   deallocate(we)
    !   deallocate(fe)
    !
    !   deallocate(kn)
    !   deallocate(ke)
    !end subroutine cleanZig

    !Generate random 32-bit integers
    function shr3( ) RESULT( ival )
       integer  ::  ival

       jz = jsr
       jsr = IEOR( jsr, ISHFT( jsr,  13 ) )
       jsr = IEOR( jsr, ISHFT( jsr, -17 ) )
       jsr = IEOR( jsr, ISHFT( jsr,   5 ) )
       ival = jz + jsr
       RETURN
    end function shr3

    !Generate uniformly distributed random numbers
    function uni( ) RESULT( fn_val )
       integer,parameter:: dpz = selected_real_kind(12,60)
       real(dpz)  ::  fn_val

       fn_val = half + 0.2328306e-9_dpz * shr3( )
       RETURN
    end function uni

    !Generate random normals
    function rnor( ) RESULT( fn_val )
       integer,parameter:: dpz = selected_real_kind(12,60)
       real(dpz)             ::  fn_val
       real(dpz), PARAMETER  ::  r = 3.442620_dpz
       real(dpz)             ::  x, y

       if( .NOT. initialized ) then
           print*, "ZIGGURAT NOT INITIALIZED. CALL ZIGSET(int(SEED)). EXITING."
       end if

       hz = shr3( )
       iz = IAND( hz, 127 )
       if( ABS( hz ) < kn(iz) ) then
          fn_val = hz * wn(iz)
       else
          do
             if( iz == 0 ) then
                do
                   x = -0.2904764_dpz* LOG( uni( ) )
                   y = -LOG( uni( ) )
                   if( y+y >= x*x ) EXIT
                end do
                fn_val = r+x
                if( hz <= 0 ) fn_val = -fn_val
                RETURN
             end if
             x = hz * wn(iz)
             if( fn(iz) + uni( )*(fn(iz-1)-fn(iz)) < exp(-half*x*x) ) then
                fn_val = x
                RETURN
             end if
             hz = shr3( )
             iz = IAND( hz, 127 )
             if( ABS( hz ) < kn(iz) ) then
                fn_val = hz * wn(iz)
                RETURN
             end if
          end do
       end if
       RETURN
    end function rnor
end module ziggurat
 
! slower than ziggurat, good for massively parallel operations.
Module Box_Muller
    implicit none
    contains
    function get_Box_Muller(mu,sig) result(r_norm)
       implicit none
       integer,parameter:: dpz = selected_real_kind(12,60)
        
       real(dpz) :: pi=4.D0*DATAN(1.D0)
       real(dpz) :: r_norm
       real(dpz) :: u1,u2
       real(dpz) ,intent(in) :: mu,sig

       !FIXME: can most likely implement a uniform dist rand for better speed here.
       call random_number(u1)
       call random_number(u2)

       !FIXME: can use linearizatoin for log and square root considering uni rand will be small
       r_norm = mu + sig*dsqrt(-2*log(1 - u1))*cos(2*pi*(1 - u2))
       return
    end function get_Box_Muller
end Module Box_Muller

Program test
    use ziggurat
    use Box_Muller
    implicit none
    integer,parameter:: dpz = selected_real_kind(12,60)
    real(dpz)  :: x,zig_actual_mean,zig_std_dev,bm_actual_mean,bm_std_dev
    real(dpz)  :: T1,T2,time_zig,time_bm
    real(dpz)  :: cum_sum_mean,cum_sum_dev
    real(dpz)  :: mean_bm,sig_bm,mean_zig !not sure how to modify zig algorithm to accomdate variable mean and std dev
    !naive solution: do a simple range mapping
    integer :: i,num_iters=1000000

    mean_bm  = 0.0
    sig_bm   = 1.0
    mean_zig = 0.0

    call random_number(x)
    x = 1 + FLOOR((100001)*x)  !choose one from m-n+1 integers
    call zigset(int(x))

    cum_sum_mean = 0
    cum_sum_dev  = 0
    do i = 0, num_iters   
       x = rnor()
       cum_sum_mean = cum_sum_mean + x
       cum_sum_dev = cum_sum_dev + (x - mean_zig)**2
    end do
    zig_actual_mean = cum_sum_mean/num_iters
    zig_std_dev = sqrt(cum_sum_dev/num_iters)

    cum_sum_mean = 0
    cum_sum_dev  = 0
    do i = 0, num_iters   
       x = get_Box_Muller(mean_bm,sig_bm)
       cum_sum_mean = cum_sum_mean + x
       cum_sum_dev = cum_sum_dev + (x - mean_bm)**2
    end do
    bm_actual_mean = cum_sum_mean/num_iters
    bm_std_dev = sqrt(cum_sum_dev/num_iters)

    call cpu_time(T1)
    do i=0,num_iters
        x = rnor()
    end do
    call CPU_TIME(T2)
    time_zig=T2-T1

    call cpu_time(T1)
    do i=0,num_iters
        x = get_Box_Muller(mean_bm,sig_bm)
    end do
    call CPU_TIME(T2)
    time_bm=T2-T1

    print*,"---------- zig test ---------- "
    print*,"std. dev.: ", zig_std_dev
    print*,"expected mean: ", mean_zig
    print*,"actual mean: ", zig_actual_mean
    print*,"seconds / 1 million : ", time_zig
    print*,"------------------------------ "
    print*,"---------- box_muller test ---------- "
    print*,"std. dev.: ", bm_std_dev
    print*,"expected mean: ", mean_bm
    print*,"actual mean: ", bm_actual_mean
    print*,"seconds / 1 million : ", time_bm
    print*,"------------------------------ "
end Program test
