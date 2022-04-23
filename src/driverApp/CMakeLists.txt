cmake_minimum_required(VERSION 3.20.0)

set(includes 
)

set(srcs 
    main.cpp
)

set_property(SOURCE ${srcs} PROPERTY LANGUAGE CUDA)

add_library(driverApp SHARED ${includes} ${srcs})
target_link_libraries(driverApp PRIVATE common)
target_link_libraries(driverApp PRIVATE sharedLibA)
target_link_libraries(driverApp PRIVATE sharedLibB)