add_library(timer
    timer.cc
)

add_library(disp
    utils.cc
    toms748.cc
    disp.cc
    secfunc.cc
)

target_link_libraries(disp
    PRIVATE
    fmt::fmt-header-only
    timer
)


add_executable(forward
    main_forward.cc
)

target_link_libraries(forward
    PRIVATE
    disp
    fmt::fmt-header-only
    toml11
)

add_executable(secfunc
    main_secfunc.cc
)

target_link_libraries(secfunc
    PRIVATE
    disp
    fmt::fmt-header-only
    toml11
    HighFive
)
