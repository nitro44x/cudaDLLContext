/**
 * @file main.cpp
 * @brief CLI driver
 */

#include "CLI11.hpp"

#include <iostream>

#include <sharedLibA/sharedLibA.hpp>
#include <sharedLibB/sharedLibB.hpp>

struct InputParams {
    size_t arraySize = 1000;
    size_t nIterations = 10;
    bool verbose = false;
};

namespace driverApp {

template <typename T>
__global__ void combine_kernel(T const *a, T const*b, T *out, size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        out[tid] += a[tid] + sin(b[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

std::shared_ptr<Array<double>> combine(std::shared_ptr<Array<double>> aArr,
                                       std::shared_ptr<Array<double>> bArr) {
    auto out = std::make_shared<Array<double>>(aArr->size());
    combine_kernel<<<256, 256>>>(aArr->data(), bArr->data(), out->data(), aArr->size());
    return out;
}
} // namespace driverApp

int main(int argc, char **argv) {
    CLI::App app{"CUDA GPU shared library demo"};

    InputParams params;
    app.add_option("-s,--size,size", params.arraySize, "Array size [elems]")
        ->default_val(params.arraySize);

    app.add_option("-n,--iterations,iterations", params.nIterations,
                   "Number of iterations to operate on the dataset")
        ->default_val(params.nIterations);

    app.add_flag("-v,--verbose", params.verbose);

    CLI11_PARSE(app, argc, argv);

    std::cout << std::endl;
    std::cout << "Array Size [elems]: " << params.arraySize << std::endl;
    std::cout << "Iterations: " << params.nIterations << std::endl;
    std::cout << "Verbose: " << params.verbose << std::endl;
    std::cout << std::endl;

    std::shared_ptr<Array<double>> combo = nullptr;
    {
        std::cout << "Creating array in sharedLibA" << std::endl;
        auto aArr = sharedLibA::createArray(params.arraySize);

        std::cout << "inPlaceprocess(x) in sharedLibA" << std::endl;
        sharedLibA::inPlaceprocess(aArr);

        std::cout << "Creating array in sharedLibB" << std::endl;
        auto bArr = sharedLibB::createArray(params.arraySize);

        std::cout << "inPlaceprocess(x) in sharedLibB" << std::endl;
        sharedLibB::inPlaceprocess(bArr);

        std::cout << "Combined in driverApp" << std::endl;
        combo = driverApp::combine(bArr, aArr);
        cudaDeviceSynchronize();
    }

    {
        std::cout << "Creating array in sharedLibA" << std::endl;
        auto aArr = sharedLibA::createArray(params.arraySize);

        std::cout << "inPlaceprocess(x) in sharedLibB" << std::endl;
        sharedLibB::inPlaceprocess(aArr);

        std::cout << "Creating array in sharedLibB" << std::endl;
        auto bArr = sharedLibB::createArray(params.arraySize);

        std::cout << "inPlaceprocess(x) in sharedLibA" << std::endl;
        sharedLibA::inPlaceprocess(bArr);

        std::cout << "Combined in sharedLibA" << std::endl;
        auto comboA = sharedLibA::combine(aArr, bArr);

        std::cout << "Combined in sharedLibB" << std::endl;
        auto comboB = sharedLibB::combine(aArr, bArr);

        cudaDeviceSynchronize();

        double normDA = 0;
        double normAB = 0;
        double normDB = 0;
        for (size_t i = 0; i < combo->size(); ++i) {
            normDA += std::sqrt((combo->at(i) - comboA->at(i)) *
                                (combo->at(i) - comboA->at(i)));
            normDB += std::sqrt((combo->at(i) - comboB->at(i)) *
                                (combo->at(i) - comboB->at(i)));
            normAB += std::sqrt((comboA->at(i) - comboB->at(i)) *
                                (comboA->at(i) - comboB->at(i)));
        }

        std::cout << "normDA = " << normDA << std::endl;
        std::cout << "normDB = " << normDB << std::endl;
        std::cout << "normAB = " << normAB << std::endl;
    }

    return 0;
}