// Look-ahead peak limiter: sliding-minimum gain computer followed by a box
// smoother of the same length, so the gain reaches the required value exactly
// when the peak leaves the delay line (no overshoot, no hard gain steps), plus
// a slow release. A final safety clamp guarantees |y| <= ceiling.
#pragma once

#include <vector>

namespace voiceanon {

class Limiter {
public:
    Limiter(double sampleRate, float ceiling = 0.891f /* -1 dBFS */);

    int latency() const { return look_ - 1; }
    float ceiling() const { return ceiling_; }

    void process(float* x, int n);

    float gainReductionDb() const;
    void reset();

private:
    int look_;
    float ceiling_;
    float releaseCoef_;
    std::vector<float> delay_, req_, minHist_;
    int pos_ = 0;
    double boxSum_;
    float gain_ = 1.0f;
    float minGainSinceRead_ = 1.0f;
};

}  // namespace voiceanon
