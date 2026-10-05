// Streaming TD-PSOLA pitch shifter with independent formant (spectral-envelope)
// scaling by grain resampling.
//
// Voiced speech: analysis marks are placed one pitch period apart (refined by
// normalised cross-correlation so they stay pitch-synchronous). A fractional
// analysis-mark pointer kappa advances by 1 / pitchRatio per output grain; the
// grain is centred at the analysis time a(kappa) (time-scale 1, no drift) and
// is a Hann-windowed copy of the input around the nearest analysis mark, read
// at a rate of
// formantRatio (> 1 compresses the grain in time, which scales the spectral
// envelope - i.e. the formants - up by formantRatio without moving F0).
// Unvoiced speech: fixed 5 ms grains, no pitch change, same envelope scaling.
//
// Grains use an asymmetric Hann window: the past side may extend up to 12 ms
// (so consecutive grains always overlap, even for low output F0), the future
// side up to 8 ms of input. Algorithmic latency is therefore fixed at
// 12 ms + 8 ms + 2 samples, independent of the parameters.
#pragma once

#include <cstdint>
#include <vector>

#include "voiceanon/pitch_tracker.h"

namespace voiceanon {

class PsolaShifter {
public:
    explicit PsolaShifter(double sampleRate);

    int latency() const { return latency_; }

    static constexpr float kMinFormantRatio = 0.8f;
    static constexpr float kMaxFormantRatio = 1.25f;

    // pitchRatio: F0 multiplier. formantRatio: envelope multiplier.
    // intonation: pitch-range scale (1 = unchanged, < 1 flattens the melody
    // around f0Ref). f0Ref <= 0 disables intonation scaling.
    // The tracker must already have been fed the same input samples.
    void process(const float* in, float* out, int n, const PitchTracker& tracker, float pitchRatio,
                 float formantRatio, float intonation, float f0Ref);

    // Effective pitch ratio of the most recently rendered voiced grain.
    float lastEffectivePitchRatio() const { return lastPitch_; }

    void reset();

private:
    struct Mark {
        int64_t pos;
        double period;
        bool voiced;
    };

    float sampleAt(int64_t i) const;
    float interp(double pos) const;
    void generateMarks(int64_t frontier, const PitchTracker& tracker);
    int64_t refine(int64_t prev, int64_t expected, double period, int64_t limit) const;
    int64_t snapToPeak(int64_t expected, double period, int64_t lowerBound, int64_t limit) const;
    // Mark by global index (clamped to the retained, existing range).
    const Mark& markAt(int64_t index) const;
    // Analysis time for a fractional mark index (interpolated / extrapolated).
    double analysisTime(double kappa) const;

    double fs_;
    int hLeftMax_, hRightInMax_, tu_, latency_, stride_;
    std::vector<float> in_, acc_, wsum_;
    int64_t mask_;
    int64_t written_ = 0;
    std::vector<Mark> marks_;
    int64_t markCount_ = 0;
    double kappa_ = 0.0;  // fractional analysis-mark index of the next synthesis grain
    float lastPitch_ = 1.0f;
};

}  // namespace voiceanon
