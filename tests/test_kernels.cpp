/**
 * @file test_kernels.cpp
 * @brief Comprehensive unit tests for CUDA kernels (setArray, addArray).
 */

#include <gtest/gtest.h>
#include <common/kernels.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>

class KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available, skipping GPU tests";
        }
    }

    void TearDown() override {
        // Synchronize and check for CUDA errors
        cudaError_t err = cudaDeviceSynchronize();
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    }

    // Helper to allocate managed memory
    template<typename T>
    T* allocateManaged(size_t count) {
        T* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, count * sizeof(T));
        EXPECT_EQ(err, cudaSuccess) << "Failed to allocate managed memory";
        return ptr;
    }

    // Helper to free managed memory
    template<typename T>
    void freeManaged(T* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// ============================================================================
// setArray Kernel Tests
// ============================================================================

TEST_F(KernelTest, SetArrayBasic) {
    const size_t size = 1000;
    const double value = 42.5;

    double* arr = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], value) << "Mismatch at index " << i;
    }

    freeManaged(arr);
}

TEST_F(KernelTest, SetArraySingleElement) {
    double* arr = allocateManaged<double>(1);

    kernels::setArray<<<1, 1>>>(arr, 99.9, 1);
    cudaDeviceSynchronize();

    EXPECT_DOUBLE_EQ(arr[0], 99.9);

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayZeroValue) {
    const size_t size = 100;
    double* arr = allocateManaged<double>(size);

    // First set to non-zero
    kernels::setArray<<<256, 256>>>(arr, 123.0, size);
    cudaDeviceSynchronize();

    // Then set to zero
    kernels::setArray<<<256, 256>>>(arr, 0.0, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], 0.0);
    }

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayNegativeValue) {
    const size_t size = 100;
    const double value = -999.123;
    double* arr = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], value);
    }

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayLarge) {
    // Test with array size larger than total threads (256*256 = 65536)
    const size_t size = 100000;
    const double value = 7.77;
    double* arr = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    // Sample check for performance
    for (size_t i = 0; i < size; i += 1000) {
        EXPECT_DOUBLE_EQ(arr[i], value) << "Mismatch at index " << i;
    }
    // Check boundaries
    EXPECT_DOUBLE_EQ(arr[0], value);
    EXPECT_DOUBLE_EQ(arr[size - 1], value);

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayFloat) {
    const size_t size = 1000;
    const float value = 3.14159f;
    float* arr = allocateManaged<float>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(arr[i], value);
    }

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayInt) {
    const size_t size = 1000;
    const int value = 42;
    int* arr = allocateManaged<int>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(arr[i], value);
    }

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayDifferentGridConfigurations) {
    const size_t size = 1000;
    const double value = 12.34;

    // Test different grid configurations
    std::vector<std::pair<int, int>> configs = {
        {1, 1},
        {1, 32},
        {1, 256},
        {32, 32},
        {128, 128},
        {256, 256}
    };

    for (const auto& config : configs) {
        double* arr = allocateManaged<double>(size);

        kernels::setArray<<<config.first, config.second>>>(arr, value, size);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < size; ++i) {
            EXPECT_DOUBLE_EQ(arr[i], value)
                << "Mismatch at index " << i
                << " with config (" << config.first << ", " << config.second << ")";
        }

        freeManaged(arr);
    }
}

TEST_F(KernelTest, SetArrayNonPowerOf2Size) {
    // Test with non-power-of-2 sizes that don't divide evenly into thread count
    std::vector<size_t> sizes = {17, 33, 127, 257, 1023, 10007};

    for (size_t size : sizes) {
        const double value = static_cast<double>(size);
        double* arr = allocateManaged<double>(size);

        kernels::setArray<<<256, 256>>>(arr, value, size);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < size; ++i) {
            EXPECT_DOUBLE_EQ(arr[i], value)
                << "Mismatch at index " << i << " with size " << size;
        }

        freeManaged(arr);
    }
}

// ============================================================================
// addArray Kernel Tests
// ============================================================================

TEST_F(KernelTest, AddArrayBasic) {
    const size_t size = 1000;
    double* a = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    // Initialize arrays
    kernels::setArray<<<256, 256>>>(a, 5.0, size);
    kernels::setArray<<<256, 256>>>(out, 3.0, size);
    cudaDeviceSynchronize();

    // Perform addition
    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(out[i], 8.0) << "Mismatch at index " << i;
    }

    // a should be unchanged
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(a[i], 5.0);
    }

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArraySingleElement) {
    double* a = allocateManaged<double>(1);
    double* out = allocateManaged<double>(1);

    a[0] = 10.0;
    out[0] = 20.0;

    kernels::addArray<<<1, 1>>>(a, out, 1);
    cudaDeviceSynchronize();

    EXPECT_DOUBLE_EQ(out[0], 30.0);

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayWithZeros) {
    const size_t size = 100;
    double* zeros = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(zeros, 0.0, size);
    kernels::setArray<<<256, 256>>>(out, 42.0, size);
    cudaDeviceSynchronize();

    kernels::addArray<<<256, 256>>>(zeros, out, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(out[i], 42.0);
    }

    freeManaged(zeros);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayNegativeValues) {
    const size_t size = 100;
    double* a = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(a, -5.0, size);
    kernels::setArray<<<256, 256>>>(out, 10.0, size);
    cudaDeviceSynchronize();

    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(out[i], 5.0);
    }

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayCancellation) {
    const size_t size = 100;
    double* a = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(a, -10.0, size);
    kernels::setArray<<<256, 256>>>(out, 10.0, size);
    cudaDeviceSynchronize();

    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(out[i], 0.0);
    }

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayLarge) {
    const size_t size = 100000;
    double* a = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(a, 1.0, size);
    kernels::setArray<<<256, 256>>>(out, 2.0, size);
    cudaDeviceSynchronize();

    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    // Sample check
    for (size_t i = 0; i < size; i += 1000) {
        EXPECT_DOUBLE_EQ(out[i], 3.0);
    }
    EXPECT_DOUBLE_EQ(out[0], 3.0);
    EXPECT_DOUBLE_EQ(out[size - 1], 3.0);

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayMultipleAdditions) {
    const size_t size = 100;
    double* a = allocateManaged<double>(size);
    double* b = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(a, 1.0, size);
    kernels::setArray<<<256, 256>>>(b, 2.0, size);
    kernels::setArray<<<256, 256>>>(out, 0.0, size);
    cudaDeviceSynchronize();

    // Add multiple arrays to out
    kernels::addArray<<<256, 256>>>(a, out, size);
    kernels::addArray<<<256, 256>>>(b, out, size);
    kernels::addArray<<<256, 256>>>(a, out, size);  // Add a again
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(out[i], 4.0);  // 0 + 1 + 2 + 1 = 4
    }

    freeManaged(a);
    freeManaged(b);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayFloat) {
    const size_t size = 1000;
    float* a = allocateManaged<float>(size);
    float* out = allocateManaged<float>(size);

    kernels::setArray<<<256, 256>>>(a, 2.5f, size);
    kernels::setArray<<<256, 256>>>(out, 1.5f, size);
    cudaDeviceSynchronize();

    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(out[i], 4.0f);
    }

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayInt) {
    const size_t size = 1000;
    int* a = allocateManaged<int>(size);
    int* out = allocateManaged<int>(size);

    kernels::setArray<<<256, 256>>>(a, 10, size);
    kernels::setArray<<<256, 256>>>(out, 20, size);
    cudaDeviceSynchronize();

    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(out[i], 30);
    }

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, AddArrayDifferentGridConfigurations) {
    const size_t size = 1000;

    std::vector<std::pair<int, int>> configs = {
        {1, 32},
        {32, 32},
        {128, 128},
        {256, 256}
    };

    for (const auto& config : configs) {
        double* a = allocateManaged<double>(size);
        double* out = allocateManaged<double>(size);

        kernels::setArray<<<config.first, config.second>>>(a, 5.0, size);
        kernels::setArray<<<config.first, config.second>>>(out, 3.0, size);
        cudaDeviceSynchronize();

        kernels::addArray<<<config.first, config.second>>>(a, out, size);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < size; ++i) {
            EXPECT_DOUBLE_EQ(out[i], 8.0)
                << "Mismatch at index " << i
                << " with config (" << config.first << ", " << config.second << ")";
        }

        freeManaged(a);
        freeManaged(out);
    }
}

TEST_F(KernelTest, AddArrayNonPowerOf2Size) {
    std::vector<size_t> sizes = {17, 33, 127, 257, 1023, 10007};

    for (size_t size : sizes) {
        double* a = allocateManaged<double>(size);
        double* out = allocateManaged<double>(size);

        kernels::setArray<<<256, 256>>>(a, 3.0, size);
        kernels::setArray<<<256, 256>>>(out, 7.0, size);
        cudaDeviceSynchronize();

        kernels::addArray<<<256, 256>>>(a, out, size);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < size; ++i) {
            EXPECT_DOUBLE_EQ(out[i], 10.0)
                << "Mismatch at index " << i << " with size " << size;
        }

        freeManaged(a);
        freeManaged(out);
    }
}

// ============================================================================
// Precision and Edge Case Tests
// ============================================================================

TEST_F(KernelTest, SetArrayVerySmallValue) {
    const size_t size = 100;
    const double value = 1e-300;
    double* arr = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], value);
    }

    freeManaged(arr);
}

TEST_F(KernelTest, SetArrayVeryLargeValue) {
    const size_t size = 100;
    const double value = 1e300;
    double* arr = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(arr, value, size);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], value);
    }

    freeManaged(arr);
}

TEST_F(KernelTest, AddArrayAccumulation) {
    // Test numerical stability with many small additions
    const size_t size = 100;
    const int numAdditions = 1000;
    double* a = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    kernels::setArray<<<256, 256>>>(a, 0.001, size);
    kernels::setArray<<<256, 256>>>(out, 0.0, size);
    cudaDeviceSynchronize();

    for (int i = 0; i < numAdditions; ++i) {
        kernels::addArray<<<256, 256>>>(a, out, size);
    }
    cudaDeviceSynchronize();

    const double expected = 0.001 * numAdditions;
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(out[i], expected, 1e-10);
    }

    freeManaged(a);
    freeManaged(out);
}

TEST_F(KernelTest, VariedValueArray) {
    // Test with different values at different positions
    const size_t size = 100;
    double* a = allocateManaged<double>(size);
    double* out = allocateManaged<double>(size);

    // Initialize with sequential values
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<double>(i);
        out[i] = static_cast<double>(size - i);
    }

    kernels::addArray<<<256, 256>>>(a, out, size);
    cudaDeviceSynchronize();

    // Each position should have sum i + (size - i) = size
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(out[i], static_cast<double>(size))
            << "Mismatch at index " << i;
    }

    freeManaged(a);
    freeManaged(out);
}
