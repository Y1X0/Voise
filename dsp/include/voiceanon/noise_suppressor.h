// Low-latency STFT noise suppressor with an integrated energy-based VAD.
//
//  * frame  ~10.7 ms (512 @ 48 kHz, 256 @ 16 kHz), hop = frame / 2
//  * sqrt-Hann analysis + synthesis windows (perfect reconstruction at 50 %)
//  * noise PSD: minimum statistics over a ~1.5 s window (8 sub-windows) +
//    speech-presence-gated recursive averaging
//  * gain: decision-directed Wiener with a floor set by the "amount" control
//
// With amount == 0 the gain is exactly 1, so the block is transparent apart
// from its fixed (frame - hop) sample delay. Latency never changes at runtime.
#pragma once

#include <vector>

#include "voiceanon/fft.h"

namespace voiceanon {

class NoiseSuppressor {
public:
    explicit NoiseSuppressor(double sampleRate);

    int hop() const { return hop_; }
    int frameSize() const { return n_; }
    int latency() const { return n_ - hop_; }

    void setAmount(float amount);  // 0 (off) .. 1 (max ~20 dB attenuation)

    // Optional fixed per-bin gain curve (bins() values) applied after the
    // suppression gain; nullptr = none. Used for spectral-envelope reshaping.
    void setShape(const float* gains);
    int bins() const { return bins_; }

    // Processes exactly hop() samples. in and out may alias.
    void processHop(const float* in, float* out);

    bool voiceActive() const { return vad_; }
    float snrDb() const { return snrDb_; }
    float noiseFloorDb() const { return noiseDb_; }

    void reset();

private:
    double fs_;
    int n_, hop_, bins_;
    Fft fft_;
    std::vector<float> window_, frame_, ola_, tmp_;
    std::vector<float> re_, im_;
    std::vector<float> shape_;
    bool hasShape_ = false;
    std::vector<float> smooth_, minimum_, noise_, gain_, prevGain_, prevPost_, gainSmoothed_;
    std::vector<float> subMin_;    // bins_ * kSubWindows stored sub-window minima
    std::vector<float> actMin_;    // running minimum of the current sub-window
    static constexpr int kSubWindows = 8;
    int subLen_ = 1, subCount_ = 0, subFilled_ = 0, subIndex_ = 0;
    int binLo_, binHi_, binHiHigh_;  // VAD bands: [lo, hi] voiced, (hi, hiHigh] fricatives
    int frames_ = 0;
    float amount_ = 0.5f;
    bool vad_ = false;
    int hang_ = 0, hangFrames_;
    float snrDb_ = 0.0f, noiseDb_ = -120.0f;
};

}  // namespace voiceanon
