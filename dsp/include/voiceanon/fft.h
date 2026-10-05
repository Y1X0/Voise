// Minimal iterative radix-2 FFT. Sizes used by the engine are small (256-1024),
// so a straightforward complex FFT on real data is fast enough and keeps the
// DSP core free of third-party dependencies.
#pragma once

#include <vector>

namespace voiceanon {

class Fft {
public:
    explicit Fft(int n);  // n must be a power of two

    int size() const { return n_; }

    // Real forward transform. re/im receive n/2 + 1 bins.
    void forwardReal(const float* in, float* re, float* im);

    // Inverse of forwardReal. Output is scaled so that inverse(forward(x)) == x.
    void inverseReal(const float* re, const float* im, float* out);

private:
    void transform(float* re, float* im, bool inverse);

    int n_;
    std::vector<int> rev_;
    std::vector<float> cos_, sin_;
    std::vector<float> workRe_, workIm_;
};

int nextPow2(int v);

}  // namespace voiceanon
