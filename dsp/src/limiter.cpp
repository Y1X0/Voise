#include "voiceanon/limiter.h"

#include <algorithm>
#include <cmath>

namespace voiceanon {

Limiter::Limiter(double sampleRate, float ceiling) : ceiling_(ceiling) {
    look_ = std::max(2, static_cast<int>(std::lround(0.0015 * sampleRate)));
    releaseCoef_ = static_cast<float>(1.0 - std::exp(-1.0 / (0.080 * sampleRate)));
    reset();
}

void Limiter::reset() {
    delay_.assign(look_, 0.0f);
    req_.assign(look_, 1.0f);
    minHist_.assign(look_, 1.0f);
    pos_ = 0;
    boxSum_ = look_;
    gain_ = 1.0f;
    minGainSinceRead_ = 1.0f;
}

void Limiter::process(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        const float in = x[i];
        const float a = std::fabs(in);
        const float required = a > ceiling_ ? ceiling_ / a : 1.0f;

        // Sliding minimum of the required gain over the look-ahead window.
        req_[pos_] = required;
        float m = 1.0f;
        for (int k = 0; k < look_; ++k) m = std::min(m, req_[k]);

        // Box smoothing of the sliding minimum (same length).
        boxSum_ += m - minHist_[pos_];
        minHist_[pos_] = m;
        const float smoothed = static_cast<float>(boxSum_ / look_);

        // Output the oldest sample in the delay line.
        const int outPos = (pos_ + 1) % look_;
        const float delayed = delay_[outPos];
        delay_[pos_] = in;
        pos_ = outPos;

        // Fast attack via the smoother, slow release.
        if (smoothed < gain_) {
            gain_ = smoothed;
        } else {
            gain_ += (smoothed - gain_) * releaseCoef_;
        }
        float y = delayed * gain_;
        y = std::min(std::max(y, -ceiling_), ceiling_);  // safety clamp
        minGainSinceRead_ = std::min(minGainSinceRead_, gain_);
        x[i] = y;
    }
    // Guard against slow floating-point drift of the running sum.
    double s = 0.0;
    for (float v : minHist_) s += v;
    boxSum_ = s;
}

float Limiter::gainReductionDb() const { return 20.0f * std::log10(std::max(gain_, 1e-6f)); }

}  // namespace voiceanon
