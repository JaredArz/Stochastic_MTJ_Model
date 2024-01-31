module joule_heating
  use MTJ_RNG_vars

  implicit none
  integer, parameter :: dp = kind(0.0d0)

  real(kind=dp) :: cstack, tau_j, fact_j

contains

  subroutine heat_device(p_p, t, T_0, T_free)

    implicit none
    integer, parameter :: dp = kind(0.0d0)

    real(kind=dp),intent(in) :: p_p, t, T_0 ! in this context, T0 is the environmental T and the device starts from there
    real(kind=dp),intent(inout) :: T_free

    T_free = T_0 + fact_j*p_p*(1.0-exp(-t/tau_j))

    return

  end subroutine heat_device

  subroutine cool_device(t, T_RT, T_0, T_free)

    implicit none
    integer, parameter :: dp = kind(0.0d0)

    real(kind=dp),intent(in) :: t, T_RT, T_0 ! in this context, T0 is the initial temperature of the device
    real(kind=dp),intent(inout) :: T_free

    T_free = T_0 + (T_RT-T_0)*(1.0-exp(-t/tau_j))

    return
  end subroutine cool_device

  subroutine comp_cstack(A)

    implicit none
    integer, parameter :: dp = kind(0.0d0)

    real(kind=dp), intent(in) :: A

    integer :: i

    cstack = 0.0

    do i=1,nLayers

       cstack = cstack + tLayer(i)*cLayer(i)*pLayer(i)

    end do

    tau_j = cstack*(tLayer(1)*tLayer(nLayers))/(tLayer(1)*k_j(2)+tLayer(nLayers)*k_j(1))
    fact_j = tLayer(1)*tLayer(nLayers)/((tLayer(1)*k_j(2)+tLayer(nLayers)*k_j(1))*A)

    deallocate(tLayer,pLayer,cLayer)


    return

  end subroutine comp_cstack

  subroutine set_layers(A)

    implicit none
    integer, parameter :: dp = kind(0.0d0)

    real(kind=dp), intent(inout) :: A

    integer :: ioerr

    open(unit = 123, file = "stack_layout.txt", action = "read", status = "old", iostat = ioerr)
    if (ioerr .eq. 0) then
       read(123,*) nLayers
       allocate(tLayer(nLayers))
       allocate(pLayer(nLayers))
       allocate(cLayer(nLayers))
       read(123,*) tLayer
       read(123,*) pLayer
       read(123,*) cLayer
       read(123,*) k_j
    else
       print*,'error opening stack layout file'
    end if

    call comp_cstack(A)

    close(123)

    return
  end subroutine set_layers

  subroutine compute_K_and_Ms(K_295, Ms_295, T_in)
    implicit none
    integer, parameter :: dp = kind(0.0d0)
    real(kind=dp), intent(in) :: T_in, K_295, Ms_295

    real(kind=dp) :: Tc, n, q, Kstar, Mstar, cm, ck

    Tc = 1453.0
    n = 1.804
    q = 1.0583
    Kstar = 4.389e5
    Mstar = 5.8077e5
    cm = Ms_295 - Mstar*(1-(295.0/Tc)**q)
    ck = K_295 - Kstar*((Ms_295/Mstar)**n)

    Ms = Mstar*(1.0-(T_in/Tc)**q) + cm
    Ki = 2.6e-9*(Kstar*((Ms/Mstar)**n) + ck)

    return
  end subroutine compute_K_and_Ms

end module joule_heating
