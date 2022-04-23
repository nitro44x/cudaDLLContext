cmake_minimum_required(VERSION 3.20.0)

set(includes 
    include/sharedLibB/sharedLibB.hpp
)

set(srcs 
    src/sharedLibB.cpp
)

set_property(SOURCE ${srcs} PROPERTY LANGUAGE CUDA)

add_library(sharedLibB SHARED ${includes} ${srcs})
target_link_libraries(sharedLibB PUBLIC common)
target_include_directories(sharedLibB PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # <prefix>/include/mylib
)