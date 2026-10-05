// Second-order IIR section (transposed direct form II) with RBJ cookbook designs.
#pragma once

#include <cmath>

namespace voiceanon {

struct Biquad {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    float z1 = 0.0f, z2 = 0.0f;

    inline float process(float x) {
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

    void reset() { z1 = z2 = 0.0f; }

    // Flush denormals / tiny state values to zero (call once per block).
    void flush() {
        if (std::fabs(z1) < 1e-20f) z1 = 0.0f;
        if (std::fabs(z2) < 1e-20f) z2 = 0.0f;
    }

    // Copy coefficients from another section but keep this section's state.
    void setCoefs(const Biquad& o) {
        b0 = o.b0; b1 = o.b1; b2 = o.b2; a1 = o.a1; a2 = o.a2;
    }

    static Biquad lowpass(double fs, double f0, double q);
    static Biquad highpass(double fs, double f0, double q);
    static Biquad bandpass(double fs, double f0, double q);  // 0 dB peak gain
    static Biquad peaking(double fs, double f0, double q, double gainDb);
    static Biquad lowShelf(double fs, double f0, double gainDb);
    static Biquad highShelf(double fs, double f0, double gainDb);
};

}  // namespace voiceanon
