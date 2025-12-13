/**
 * @file test_array.cpp
 * @brief Comprehensive unit tests for the Array<T> template class.
 */

#include <gtest/gtest.h>
#include <common/Array.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

class ArrayTest : public ::testing::Test {
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
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(ArrayTest, DefaultConstructor) {
    Array<double> arr;
    EXPECT_EQ(arr.size(), 0);
    EXPECT_EQ(arr.nBytes(), 0);
    EXPECT_EQ(arr.data(), nullptr);
}

TEST_F(ArrayTest, SizeConstructor) {
    const size_t size = 1000;
    Array<double> arr(size);

    EXPECT_EQ(arr.size(), size);
    EXPECT_EQ(arr.nBytes(), size * sizeof(double));
    EXPECT_NE(arr.data(), nullptr);
}

TEST_F(ArrayTest, SizeConstructorSingleElement) {
    Array<double> arr(1);

    EXPECT_EQ(arr.size(), 1);
    EXPECT_EQ(arr.nBytes(), sizeof(double));
    EXPECT_NE(arr.data(), nullptr);
}

TEST_F(ArrayTest, SizeConstructorLargeArray) {
    const size_t size = 1000000;  // 1 million elements
    Array<double> arr(size);

    EXPECT_EQ(arr.size(), size);
    EXPECT_EQ(arr.nBytes(), size * sizeof(double));
    EXPECT_NE(arr.data(), nullptr);
}

TEST_F(ArrayTest, ValueConstructor) {
    const size_t size = 100;
    const double initValue = 42.5;
    Array<double> arr(size, initValue);

    cudaDeviceSynchronize();

    EXPECT_EQ(arr.size(), size);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), initValue) << "Mismatch at index " << i;
    }
}

TEST_F(ArrayTest, ValueConstructorZeroValue) {
    const size_t size = 100;
    Array<double> arr(size, 0.0);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), 0.0) << "Mismatch at index " << i;
    }
}

TEST_F(ArrayTest, ValueConstructorNegativeValue) {
    const size_t size = 100;
    const double initValue = -123.456;
    Array<double> arr(size, initValue);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), initValue) << "Mismatch at index " << i;
    }
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(ArrayTest, MoveConstructor) {
    const size_t size = 100;
    const double initValue = 99.9;
    Array<double> original(size, initValue);

    cudaDeviceSynchronize();

    double* originalPtr = original.data();
    Array<double> moved(std::move(original));

    // Moved-to array should have the data
    EXPECT_EQ(moved.size(), size);
    EXPECT_EQ(moved.data(), originalPtr);

    // Moved-from array should be empty
    EXPECT_EQ(original.size(), 0);

    // Verify data integrity
    cudaDeviceSynchronize();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(moved.at(i), initValue);
    }
}

TEST_F(ArrayTest, MoveAssignment) {
    const size_t size1 = 100;
    const size_t size2 = 200;
    const double value1 = 1.0;
    const double value2 = 2.0;

    Array<double> arr1(size1, value1);
    Array<double> arr2(size2, value2);

    cudaDeviceSynchronize();

    double* arr2Ptr = arr2.data();
    arr1 = std::move(arr2);

    // arr1 should now have arr2's data
    EXPECT_EQ(arr1.size(), size2);
    EXPECT_EQ(arr1.data(), arr2Ptr);

    // arr2 should be empty
    EXPECT_EQ(arr2.size(), 0);

    // Verify data integrity
    cudaDeviceSynchronize();
    for (size_t i = 0; i < size2; ++i) {
        EXPECT_DOUBLE_EQ(arr1.at(i), value2);
    }
}

TEST_F(ArrayTest, MoveAssignmentSelfAssign) {
    const size_t size = 100;
    const double value = 42.0;
    Array<double> arr(size, value);

    cudaDeviceSynchronize();

    double* originalPtr = arr.data();
    arr = std::move(arr);  // Self-assignment

    // Should remain unchanged
    EXPECT_EQ(arr.size(), size);
    EXPECT_EQ(arr.data(), originalPtr);
}

// ============================================================================
// setValue Tests
// ============================================================================

TEST_F(ArrayTest, SetValue) {
    const size_t size = 1000;
    Array<double> arr(size);

    arr.setValue(3.14159);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), 3.14159) << "Mismatch at index " << i;
    }
}

TEST_F(ArrayTest, SetValueMultipleTimes) {
    const size_t size = 100;
    Array<double> arr(size);

    // First set
    arr.setValue(1.0);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), 1.0);
    }

    // Second set - different value
    arr.setValue(2.0);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), 2.0);
    }

    // Third set - zero
    arr.setValue(0.0);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), 0.0);
    }
}

TEST_F(ArrayTest, SetValueLargeArray) {
    const size_t size = 100000;  // Large enough to span multiple thread blocks
    const double value = 7.77;
    Array<double> arr(size);

    arr.setValue(value);
    cudaDeviceSynchronize();

    // Sample check (checking all would be slow)
    for (size_t i = 0; i < size; i += 1000) {
        EXPECT_DOUBLE_EQ(arr.at(i), value) << "Mismatch at index " << i;
    }
    // Check boundaries
    EXPECT_DOUBLE_EQ(arr.at(0), value);
    EXPECT_DOUBLE_EQ(arr.at(size - 1), value);
}

// ============================================================================
// operator+= Tests
// ============================================================================

TEST_F(ArrayTest, InPlaceAddition) {
    const size_t size = 1000;
    const double value1 = 5.0;
    const double value2 = 3.0;

    Array<double> arr1(size, value1);
    Array<double> arr2(size, value2);

    cudaDeviceSynchronize();

    arr1 += arr2;
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr1.at(i), value1 + value2) << "Mismatch at index " << i;
    }

    // arr2 should be unchanged
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr2.at(i), value2);
    }
}

TEST_F(ArrayTest, InPlaceAdditionMultiple) {
    const size_t size = 100;
    Array<double> arr1(size, 1.0);
    Array<double> arr2(size, 2.0);
    Array<double> arr3(size, 3.0);

    cudaDeviceSynchronize();

    arr1 += arr2;
    arr1 += arr3;
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr1.at(i), 6.0);  // 1 + 2 + 3
    }
}

TEST_F(ArrayTest, InPlaceAdditionWithZeros) {
    const size_t size = 100;
    const double value = 42.0;

    Array<double> arr1(size, value);
    Array<double> zeros(size, 0.0);

    cudaDeviceSynchronize();

    arr1 += zeros;
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr1.at(i), value);
    }
}

TEST_F(ArrayTest, InPlaceAdditionNegativeValues) {
    const size_t size = 100;

    Array<double> arr1(size, 10.0);
    Array<double> arr2(size, -5.0);

    cudaDeviceSynchronize();

    arr1 += arr2;
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr1.at(i), 5.0);
    }
}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST_F(ArrayTest, AtConstAccess) {
    const size_t size = 10;
    const double value = 123.456;
    const Array<double> arr(size, value);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), value);
    }
}

TEST_F(ArrayTest, AtMutableAccess) {
    const size_t size = 10;
    Array<double> arr(size, 0.0);

    cudaDeviceSynchronize();

    // Modify via at()
    for (size_t i = 0; i < size; ++i) {
        arr.at(i) = static_cast<double>(i);
    }

    // Verify modifications
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), static_cast<double>(i));
    }
}

TEST_F(ArrayTest, DataPointerAccess) {
    const size_t size = 10;
    const double value = 99.0;
    Array<double> arr(size, value);

    cudaDeviceSynchronize();

    double* ptr = arr.data();
    EXPECT_NE(ptr, nullptr);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(ptr[i], value);
    }
}

TEST_F(ArrayTest, ConstDataPointerAccess) {
    const size_t size = 10;
    const double value = 88.0;
    const Array<double> arr(size, value);

    cudaDeviceSynchronize();

    const double* ptr = arr.data();
    EXPECT_NE(ptr, nullptr);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(ptr[i], value);
    }
}

// ============================================================================
// Size and Bytes Tests
// ============================================================================

TEST_F(ArrayTest, SizeMethod) {
    Array<double> arr1(100);
    EXPECT_EQ(arr1.size(), 100);

    Array<double> arr2(1);
    EXPECT_EQ(arr2.size(), 1);

    Array<double> arr3;
    EXPECT_EQ(arr3.size(), 0);
}

TEST_F(ArrayTest, NBytesMethod) {
    Array<double> arr1(100);
    EXPECT_EQ(arr1.nBytes(), 100 * sizeof(double));

    Array<float> arr2(100);
    EXPECT_EQ(arr2.nBytes(), 100 * sizeof(float));

    Array<int> arr3(100);
    EXPECT_EQ(arr3.nBytes(), 100 * sizeof(int));
}

// ============================================================================
// Different Data Type Tests
// ============================================================================

TEST_F(ArrayTest, FloatArray) {
    const size_t size = 100;
    const float value = 3.14f;
    Array<float> arr(size, value);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(arr.at(i), value);
    }
}

TEST_F(ArrayTest, IntArray) {
    const size_t size = 100;
    const int value = 42;
    Array<int> arr(size, value);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(arr.at(i), value);
    }
}

TEST_F(ArrayTest, IntArrayAddition) {
    const size_t size = 100;
    Array<int> arr1(size, 10);
    Array<int> arr2(size, 20);

    cudaDeviceSynchronize();

    arr1 += arr2;
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(arr1.at(i), 30);
    }
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(ArrayTest, VerySmallValues) {
    const size_t size = 10;
    const double value = 1e-300;
    Array<double> arr(size, value);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), value);
    }
}

TEST_F(ArrayTest, VeryLargeValues) {
    const size_t size = 10;
    const double value = 1e300;
    Array<double> arr(size, value);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr.at(i), value);
    }
}

TEST_F(ArrayTest, ChainedOperations) {
    const size_t size = 100;
    Array<double> arr1(size, 1.0);
    Array<double> arr2(size, 2.0);
    Array<double> arr3(size, 3.0);

    cudaDeviceSynchronize();

    // Chain operations
    (arr1 += arr2) += arr3;
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(arr1.at(i), 6.0);
    }
}

TEST_F(ArrayTest, MultipleArrayLifecycles) {
    // Test that multiple arrays can be created and destroyed without issues
    for (int i = 0; i < 10; ++i) {
        Array<double> arr(1000, static_cast<double>(i));
        cudaDeviceSynchronize();
        EXPECT_DOUBLE_EQ(arr.at(0), static_cast<double>(i));
    }
}
