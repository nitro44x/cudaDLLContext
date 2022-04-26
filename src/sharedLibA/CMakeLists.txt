cmake_minimum_required(VERSION 3.20.0)

set(includes 
    include/sharedLibA/export.hpp
    include/sharedLibA/sharedLibA.hpp
)

set(srcs 
    src/force.cpp
)

set(cuda_srcs 
    src/sharedLibA.cpp
)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)

add_library(sharedLibA SHARED ${includes} ${srcs} ${cuda_srcs})
target_link_libraries(sharedLibA PUBLIC common)
target_include_directories(sharedLibA PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # <prefix>/include/mylib
)