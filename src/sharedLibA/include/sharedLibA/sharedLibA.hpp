#pragma once

#include <sharedLibA/export.hpp>

#include <common/Array.hpp>

#include <memory>

namespace sharedLibA {

API std::shared_ptr<Array<double>> createArray(size_t size);

API void inPlaceprocess(std::shared_ptr<Array<double>> a);

API std::shared_ptr<Array<double>> combine(std::shared_ptr<Array<double>> aArr,
                                           std::shared_ptr<Array<double>> bArr);

} // namespace sharedLibA
