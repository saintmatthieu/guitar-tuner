#pragma once

#include <complex>
#include <vector>

#include "Utils.h"
#include "pffft.h"

namespace saint {
// PFFT memory alignment requirement
template <typename T>
struct alignas(16) Aligned {
    T value;
};

class RealFft {
   public:
    RealFft(int size) : setup(pffft_new_setup(size, PFFFT_REAL)), size(size) {
        assert(utils::isPowerOfTwo(size));
        work.value.resize(size);
    }

    ~RealFft() {
        if (setup)
            pffft_destroy_setup(setup);
    }

    RealFft(const RealFft&) = delete;
    RealFft& operator=(const RealFft&) = delete;

    RealFft(RealFft&& other) noexcept
        : setup(other.setup), size(other.size), work(std::move(other.work)) {
        other.setup = nullptr;
    }

    RealFft& operator=(RealFft&& other) noexcept {
        if (this != &other) {
            if (setup)
                pffft_destroy_setup(setup);
            setup = other.setup;
            size = other.size;
            work = std::move(other.work);
            other.setup = nullptr;
        }
        return *this;
    }

    void forward(const float* input, float* output) {
        pffft_transform_ordered(setup, input, output, work.value.data(), PFFFT_FORWARD);
    }

    void forward(const float* input, std::complex<float>* output) {
        pffft_transform_ordered(setup, input, reinterpret_cast<float*>(output), work.value.data(),
                                PFFFT_FORWARD);
    }

    void inverse(const float* input, float* output) {
        pffft_transform_ordered(setup, input, output, work.value.data(), PFFFT_BACKWARD);
    }

    void inverse(const std::complex<float>* input, float* output) {
        pffft_transform_ordered(setup, reinterpret_cast<const float*>(input), output,
                                work.value.data(), PFFFT_BACKWARD);
    }

    PFFFT_Setup* setup = nullptr;
    int size = 0;

   private:
    Aligned<std::vector<float>> work;
};
}  // namespace saint