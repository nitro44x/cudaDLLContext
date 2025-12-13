cmake_minimum_required(VERSION 3.20.0)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test sources (CUDA files)
set(test_sources
    test_array.cpp
    test_kernels.cpp
)

# Mark test sources as CUDA files since they use CUDA code
set_property(SOURCE ${test_sources} PROPERTY LANGUAGE CUDA)

# Create test executable
add_executable(unit_tests ${test_sources})

target_link_libraries(unit_tests
    PRIVATE
        common
        GTest::gtest_main
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/include
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)
