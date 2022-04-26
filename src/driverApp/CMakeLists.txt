cmake_minimum_required(VERSION 3.20.0)

set(includes 
    CLI11.hpp
)

set(srcs 
    force.cpp
)

set(cuda_srcs 
    main.cpp
)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)

add_executable(driverApp ${includes} ${srcs} ${cuda_srcs})
target_link_libraries(driverApp PRIVATE common)
target_link_libraries(driverApp PRIVATE sharedLibA)
target_link_libraries(driverApp PRIVATE sharedLibB)

add_custom_command(TARGET driverApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:sharedLibA> $<TARGET_FILE_DIR:driverApp>
)

add_custom_command(TARGET driverApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:sharedLibB> $<TARGET_FILE_DIR:driverApp>
)