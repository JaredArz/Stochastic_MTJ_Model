module glo
    implicit none
    integer :: a = 1
    contains
        subroutine comp(val)
            implicit none
            integer, intent(inout) :: val
            val = val + a
        end subroutine
end module glo

program mains
    use glo
    use ziggurat

    implicit none
    integer,parameter:: dpz = selected_real_kind(12,60)
    real(dpz)  :: T1,T2,run_time
    integer :: i,x,num_iters=1000000

        call random_number(seed)
        !FIXME: get an int please
        call zigset(int(1+floor((1000001)*seed)))
    call cpu_time(T1)
    do i = 0, num_iters   
        call comp(x)
    end do

    call CPU_TIME(T2)
    run_time=T2-T1
    print*,x

    print*,"---------- sample ---------- "
    print*,"seconds / 1 million : ", run_time
end program mains
