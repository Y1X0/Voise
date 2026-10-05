#include "voiceanon/fft.h"

#include <cmath>
#include <stdexcept>

namespace voiceanon {

int nextPow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

Fft::Fft(int n) : n_(n) {
    if (n < 2 || (n & (n - 1)) != 0) throw std::invalid_argument("FFT size must be a power of two");
    rev_.resize(n);
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        rev_[i] = r;
    }
    cos_.resize(n / 2);
    sin_.resize(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        const double a = -2.0 * M_PI * i / n;
        cos_[i] = static_cast<float>(std::cos(a));
        sin_[i] = static_cast<float>(std::sin(a));
    }
    workRe_.resize(n);
    workIm_.resize(n);
}

void Fft::transform(float* re, float* im, bool inverse) {
    for (int i = 0; i < n_; ++i) {
        const int j = rev_[i];
        if (j > i) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n_; len <<= 1) {
        const int half = len >> 1;
        const int step = n_ / len;
        for (int start = 0; start < n_; start += len) {
            for (int k = 0; k < half; ++k) {
                const float wr = cos_[k * step];
                const float wi = inverse ? -sin_[k * step] : sin_[k * step];
                const int a = start + k;
                const int b = a + half;
                const float tr = re[b] * wr - im[b] * wi;
                const float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }
}

void Fft::forwardReal(const float* in, float* re, float* im) {
    for (int i = 0; i < n_; ++i) {
        workRe_[i] = in[i];
        workIm_[i] = 0.0f;
    }
    transform(workRe_.data(), workIm_.data(), false);
    for (int i = 0; i <= n_ / 2; ++i) {
        re[i] = workRe_[i];
        im[i] = workIm_[i];
    }
}

void Fft::inverseReal(const float* re, const float* im, float* out) {
    const int h = n_ / 2;
    for (int i = 0; i <= h; ++i) {
        workRe_[i] = re[i];
        workIm_[i] = im[i];
    }
    for (int i = h + 1; i < n_; ++i) {  // Hermitian symmetry
        workRe_[i] = re[n_ - i];
        workIm_[i] = -im[n_ - i];
    }
    transform(workRe_.data(), workIm_.data(), true);
    const float scale = 1.0f / static_cast<float>(n_);
    for (int i = 0; i < n_; ++i) out[i] = workRe_[i] * scale;
}

}  // namespace voiceanon
