cmake_minimum_required(VERSION 3.20.0)

set(includes 
    include/common/export.hpp
    include/common/Array.hpp
    include/common/kernels.hpp
    include/common/oversubTest.hpp
)

set(srcs 
    src/force.cpp
)

set(cuda_srcs 
    src/Array.cpp
    src/kernels.cpp
    src/oversubTest.cpp
)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)

add_library(common STATIC ${includes} ${srcs} ${cuda_srcs})
target_include_directories(common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # <prefix>/include/mylib
)