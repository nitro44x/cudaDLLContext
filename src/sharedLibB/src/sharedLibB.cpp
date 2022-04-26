#include <sharedLibB/sharedLibB.hpp>

namespace sharedLibB {

std::shared_ptr<Array<double>> createArray(size_t size) {
    return std::make_shared<Array<double>>(size);
}

void inPlaceprocess(std::shared_ptr<Array<double>> a) { a->setValue(123.321); }

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
    auto out = createArray(aArr->size());
    combine_kernel<<<256, 256>>>(aArr->data(), bArr->data(), out->data(), aArr->size());
    return out;
}

} // namespace sharedLibB