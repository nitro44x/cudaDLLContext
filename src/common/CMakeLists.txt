cmake_minimum_required(VERSION 3.20.0)

set(includes 
    include/common/Array.hpp
    include/common/kernels.hpp
    include/common/oversubTests.hpp
)

set(srcs 
    src/Array.cpp
    src/kernels.cpp
    src/oversubTests.cpp
)

set_property(SOURCE ${srcs} PROPERTY LANGUAGE CUDA)

add_library(common SHARED ${includes} ${srcs})
target_include_directories(common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # <prefix>/include/mylib
)