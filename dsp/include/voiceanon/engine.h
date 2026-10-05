// Real-time voice anonymization engine (speaker-identity obfuscation).
//
// Streaming pipeline (all stages run in a fixed internal quantum, so output is
// bit-identical regardless of the caller's block size):
//
//   input -> sanitize / loss concealment -> high-pass 70 Hz
//         -> STFT noise suppression + VAD
//         -> YIN pitch tracking
//         -> TD-PSOLA pitch shift + formant (envelope) scaling + intonation scaling
//         -> spectral colour (tilt shelves, clarity EQ)
//         -> AGC (speech-gated) + residual gate + output gain
//         -> look-ahead limiter -> output
//
// Thread model: process() runs on the audio thread and never allocates or locks.
// setParams()/metrics() may be called from any thread (lock-free atomics).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "voiceanon/biquad.h"
#include "voiceanon/limiter.h"
#include "voiceanon/noise_suppressor.h"
#include "voiceanon/pitch_tracker.h"
#include "voiceanon/psola.h"

namespace voiceanon {

enum class Direction : int { Auto = 0, Up = 1, Down = -1 };

struct Params {
    bool anonymize = true;         // false = latency-matched dry bypass (A/B testing only)
    float pitchSemitones = 3.0f;   // F0 shift for the "Up" direction (sign flips for "Down")
    float formantPercent = 10.0f;  // spectral-envelope shift for "Up" (+10 % => x1.10)
    float intonation = 0.85f;      // pitch-range scale, 1 = keep melody, < 1 = flatter
    float tiltDb = 1.5f;           // spectral tilt change (identity cue) for "Up"
    float clarity = 0.5f;          // 0..1 presence boost / low-mid cleanup
    float noiseSuppression = 0.5f; // 0..1
    float outputGainDb = 0.0f;     // -12..+12
    bool agc = true;               // speech-gated level normalisation
    Direction direction = Direction::Auto;
    // Auto mode only: direction to use until enough voiced speech has been heard
    // to decide (e.g. the direction remembered from the previous session).
    // 0 = none (defaults to Up), +1 Up, -1 Down.
    int autoDirectionHint = 0;

    // --- Experimental dimensions (all 0 = off; presets leave them off) ---------
    float spectralReshapeDb = 0.0f;     // smooth random spectral-envelope reshaping, +-dB
    float pitchDriftSemitones = 0.0f;   // slow random F0 offset (controlled prosody), +-st
    float formantJitterPercent = 0.0f;  // slow random variation of the envelope warp over time, +-%
    float dynamicsFlatten = 0.0f;       // 0..1 syllable-rate energy-contour compression
    uint32_t variationSeed = 1;         // seed for the random processes above
};

struct Metrics {
    float inputRmsDb = -120.0f, inputPeakDb = -120.0f;
    float outputRmsDb = -120.0f, outputPeakDb = -120.0f;
    float f0In = 0.0f;           // current input F0 (0 = unvoiced)
    float f0OutEstimate = 0.0f;  // f0In * effective pitch ratio
    float meanF0 = 0.0f;         // long-term speaker F0 estimate
    bool voiced = false;
    bool vad = false;
    float pitchRatio = 1.0f;
    float formantRatio = 1.0f;
    int direction = 0;  // +1 up, -1 down, 0 undecided (auto)
    float noiseFloorDb = -120.0f;
    float limiterGainDb = 0.0f;
    float agcGainDb = 0.0f;
    uint64_t lostFrames = 0;
    uint64_t sanitizedSamples = 0;
    uint64_t processedFrames = 0;
    float latencyMs = 0.0f;
};

class Engine {
public:
    explicit Engine(double sampleRate, int maxBlock = 4096);

    // Thread-safe parameter update (applied with smoothing at the next quantum).
    void setParams(const Params& p);
    Params params() const;

    // Audio thread. in == nullptr signals that n input frames were lost
    // (e.g. capture underrun); the engine conceals them smoothly.
    void process(const float* in, float* out, int n);

    int latencySamples() const { return latency_; }
    double latencyMs() const { return 1000.0 * latency_ / fs_; }
    double sampleRate() const { return fs_; }
    int quantum() const { return hop_; }

    Metrics metrics() const;

    // Audio thread (or while not running).
    void reset();

    // --- In-memory A/B capture for the on-device test mode (never written to disk
    // by the engine). startCapture() allocates and must not be called on the
    // audio thread. Captured dry signal is latency-aligned with the wet signal.
    bool startCapture(double seconds);
    bool captureReady() const { return captureState_.load(std::memory_order_acquire) == 2; }
    float captureProgress() const;
    // Copies the capture out and re-arms the idle state. Returns sample count.
    int readCapture(std::vector<float>& dry, std::vector<float>& wet);

private:
    void processQuantum(const float* in, float* out);
    void updateSmoothedParams(bool voicedNow, float f0Now, bool vadNow);
    void updateEq(bool force);

    struct AtomicParams {
        std::atomic<bool> anonymize{true};
        std::atomic<float> pitchSemitones{3.0f}, formantPercent{10.0f}, intonation{0.85f}, tiltDb{1.5f},
            clarity{0.5f}, noiseSuppression{0.5f}, outputGainDb{0.0f};
        std::atomic<bool> agc{true};
        std::atomic<int> direction{0};
        std::atomic<int> autoHint{0};
        std::atomic<float> reshapeDb{0}, driftSt{0}, fjitterPct{0}, flatten{0};
        std::atomic<uint32_t> seed{1};
    } ap_;

    struct AtomicMetrics {
        std::atomic<float> inRms{-120}, inPeak{-120}, outRms{-120}, outPeak{-120}, f0In{0}, f0Out{0}, meanF0{0},
            pitchRatio{1}, formantRatio{1}, noiseFloor{-120}, limiterGain{0}, agcGain{0};
        std::atomic<bool> voiced{false}, vad{false};
        std::atomic<int> direction{0};
        std::atomic<uint64_t> lost{0}, sanitized{0}, processed{0};
    } am_;

    double fs_;
    int hop_, maxBlock_;
    int latency_;

    // FIFOs that decouple the caller's block size from the internal quantum.
    std::vector<float> inFifo_, outFifo_;
    int inFill_ = 0;
    int outRead_ = 0, outFill_ = 0;
    std::vector<float> qIn_, qOut_, work_, work2_;

    // Stages
    Biquad hpf_;
    NoiseSuppressor ns_;
    PitchTracker tracker_;
    PsolaShifter psola_;
    Biquad lowShelf_, highShelf_, presence_, mud_;
    Limiter limiter_;

    // Dry (bypass) path delay = wet latency inside a quantum.
    std::vector<float> dryDelay_;
    int dryPos_ = 0, dryDelayLen_;

    // Smoothed parameter state (audio thread only)
    float sPitchLog2_ = 0.0f, sFormantLog2_ = 0.0f, sIntonation_ = 1.0f, sTilt_ = 0.0f, sClarity_ = 0.0f;
    float sOutGain_ = 1.0f, mix_ = 1.0f;
    float eqTilt_ = 1e9f, eqClarity_ = 1e9f;
    float smoothCoef_;
    int latchedDirection_ = 0;
    double voicedSeconds_ = 0.0, logF0Mean_ = 0.0;
    float agcGainDb_ = 0.0f, rmsEnv_ = 0.0f, gateGain_ = 1.0f;
    int silentQuanta_ = 0;
    float lastInput_ = 0.0f;
    int fadeInRemaining_ = 0, fadeLen_;

    // Experimental processes
    void updateReshape(float db, uint32_t seed);
    float nextRandom();  // uniform [-1, 1)
    std::vector<float> shape_;
    float shapeDb_ = 0.0f;
    uint32_t shapeSeed_ = 0, rng_ = 1, rngSeed_ = 0;
    float driftTarget_ = 0.0f, drift_ = 0.0f, fjTarget_ = 0.0f, fj_ = 0.0f;
    int driftCountdown_ = 0, fjCountdown_ = 0;
    float dynEnv_ = 0.0f;

    // Capture
    std::vector<float> capDry_, capWet_;
    std::atomic<int> captureState_{0};  // 0 idle, 1 recording, 2 ready
    std::atomic<int> capPos_{0};
    int capLen_ = 0;
};

}  // namespace voiceanon
