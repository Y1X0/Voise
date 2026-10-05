// Streaming YIN pitch tracker running on a decimated (~12 kHz), low-passed
// copy of the signal. Range 65-500 Hz. Updated every ~5 ms.
#pragma once

#include <cstdint>
#include <vector>

#include "voiceanon/biquad.h"

namespace voiceanon {

class PitchTracker {
public:
    explicit PitchTracker(double sampleRate);

    void push(const float* x, int n);

    bool voiced() const { return voiced_; }
    // Period in full-rate samples (valid when voiced()).
    double periodSamples() const { return period_; }
    float f0() const { return voiced_ ? static_cast<float>(fs_ / period_) : 0.0f; }
    // 1 - (normalised YIN dip), 0..1.
    float confidence() const { return confidence_; }

    void reset();

    static constexpr double kMinF0 = 65.0;
    static constexpr double kMaxF0 = 500.0;

private:
    void analyze();

    double fs_, fsd_;
    int dec_, decPhase_ = 0;
    Biquad lp1_, lp2_;
    std::vector<float> ring_;
    int mask_;
    int64_t written_ = 0;
    int tauMin_, tauMax_, window_, hop_, sinceLast_ = 0;
    std::vector<float> buf_, diff_;
    bool voiced_ = false;
    double period_ = 0.0;
    float confidence_ = 0.0f;
};

}  // namespace voiceanon
