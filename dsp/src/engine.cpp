#include "voiceanon/engine.h"

#include <algorithm>
#include <cmath>

namespace voiceanon {
namespace {

constexpr float kAgcTargetDb = -20.0f;
constexpr float kAgcMinDb = -6.0f;
constexpr float kAgcMaxDb = 12.0f;
constexpr double kAutoF0Split = 165.0;  // Hz: below -> shift up, above -> shift down
constexpr double kDirectionLatchSeconds = 0.5;  // voiced speech needed to decide Auto direction

inline float toDb(float power) { return 10.0f * std::log10(power + 1e-12f); }
inline float clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

}  // namespace

Engine::Engine(double sampleRate, int maxBlock)
    : fs_(sampleRate),
      maxBlock_(std::max(1, maxBlock)),
      ns_(sampleRate),
      tracker_(sampleRate),
      psola_(sampleRate),
      limiter_(sampleRate) {
    hop_ = ns_.hop();
    hpf_ = Biquad::highpass(fs_, 70.0, 0.7071);
    inFifo_.assign(hop_, 0.0f);
    outFifo_.assign(2 * hop_, 0.0f);
    qIn_.assign(hop_, 0.0f);
    qOut_.assign(hop_, 0.0f);
    work_.assign(hop_, 0.0f);
    work2_.assign(hop_, 0.0f);
    dryDelayLen_ = ns_.latency() + psola_.latency() + limiter_.latency();
    dryDelay_.assign(std::max(1, dryDelayLen_), 0.0f);
    latency_ = (hop_ - 1) + dryDelayLen_;
    smoothCoef_ = static_cast<float>(1.0 - std::exp(-hop_ / (0.08 * fs_)));
    fadeLen_ = std::max(1, static_cast<int>(0.002 * fs_));
    shape_.assign(ns_.bins(), 1.0f);
    reset();
}

void Engine::reset() {
    hpf_.reset();
    ns_.reset();
    tracker_.reset();
    psola_.reset();
    limiter_.reset();
    lowShelf_.reset();
    highShelf_.reset();
    presence_.reset();
    mud_.reset();
    std::fill(dryDelay_.begin(), dryDelay_.end(), 0.0f);
    dryPos_ = 0;
    inFill_ = 0;
    std::fill(outFifo_.begin(), outFifo_.end(), 0.0f);
    outRead_ = 0;
    outFill_ = hop_ - 1;  // pre-fill so any caller block size can be served
    const Params p = params();
    const float sign = (p.direction == Direction::Down ||
                        (p.direction == Direction::Auto && p.autoDirectionHint < 0)) ? -1.0f : 1.0f;
    sPitchLog2_ = sign * clampf(p.pitchSemitones, -7.0f, 7.0f) / 12.0f;
    {
        float r = 1.0f + clampf(p.formantPercent, -20.0f, 25.0f) / 100.0f;
        if (sign < 0) r = 1.0f / r;
        sFormantLog2_ = std::log2(clampf(r, PsolaShifter::kMinFormantRatio, PsolaShifter::kMaxFormantRatio));
    }
    sIntonation_ = clampf(p.intonation, 0.5f, 1.5f);
    sTilt_ = sign * clampf(p.tiltDb, -6.0f, 6.0f);
    sClarity_ = clampf(p.clarity, 0.0f, 1.0f);
    sOutGain_ = std::pow(10.0f, clampf(p.outputGainDb, -12.0f, 12.0f) / 20.0f);
    mix_ = p.anonymize ? 1.0f : 0.0f;
    latchedDirection_ = 0;
    voicedSeconds_ = 0.0;
    logF0Mean_ = 0.0;
    agcGainDb_ = 0.0f;
    rmsEnv_ = 0.0f;
    gateGain_ = 1.0f;
    silentQuanta_ = 0;
    lastInput_ = 0.0f;
    fadeInRemaining_ = 0;
    eqTilt_ = eqClarity_ = 1e9f;
    updateEq(true);
    rngSeed_ = p.variationSeed;
    rng_ = p.variationSeed ? p.variationSeed * 2654435761u + 12345u : 12345u;
    driftTarget_ = drift_ = fjTarget_ = fj_ = 0.0f;
    driftCountdown_ = fjCountdown_ = 0;
    dynEnv_ = 0.0f;
    shapeDb_ = 0.0f;
    shapeSeed_ = 0;
    ns_.setShape(nullptr);
    updateReshape(p.spectralReshapeDb, p.variationSeed);
}

float Engine::nextRandom() {
    // xorshift32
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return static_cast<float>(rng_ & 0xFFFFFF) / static_cast<float>(0x800000) - 1.0f;
}

void Engine::updateReshape(float db, uint32_t seed) {
    if (db == shapeDb_ && seed == shapeSeed_) return;
    shapeDb_ = db;
    shapeSeed_ = seed;
    if (db <= 0.0f) {
        ns_.setShape(nullptr);
        return;
    }
    // Smooth random gain curve over a mel axis: 8 control points 150 Hz..min(7 kHz, fs/2),
    // values in [-db, +db], cosine-interpolated, mean-normalised over the speech band.
    uint32_t r = seed * 747796405u + 2891336453u;
    auto rnd = [&r]() {
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        return static_cast<float>(r & 0xFFFFFF) / static_cast<float>(0x800000) - 1.0f;
    };
    constexpr int kPoints = 8;
    float ctrl[kPoints + 2];
    for (int i = 0; i < kPoints + 2; ++i) ctrl[i] = rnd() * db;
    ctrl[0] = ctrl[kPoints + 1] = 0.0f;  // no change at the extremes
    auto mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    const double lo = mel(150.0), hi = mel(std::min(7000.0, 0.45 * fs_));
    const int bins = ns_.bins();
    const double binHz = fs_ / (2.0 * (bins - 1));
    double sum = 0.0;
    int count = 0;
    for (int k = 0; k < bins; ++k) {
        const double m = (mel(k * binHz) - lo) / (hi - lo) * (kPoints + 1);
        float g;
        if (m <= 0.0) g = ctrl[0];
        else if (m >= kPoints + 1) g = ctrl[kPoints + 1];
        else {
            const int i = static_cast<int>(m);
            const double t = 0.5 - 0.5 * std::cos(M_PI * (m - i));
            g = static_cast<float>(ctrl[i] * (1.0 - t) + ctrl[i + 1] * t);
        }
        shape_[k] = g;
        if (k * binHz > 300.0 && k * binHz < 4000.0) {
            sum += g;
            ++count;
        }
    }
    const float mean = count ? static_cast<float>(sum / count) : 0.0f;
    for (int k = 0; k < bins; ++k) shape_[k] = std::pow(10.0f, (shape_[k] - mean) / 20.0f);
    ns_.setShape(shape_.data());
}

void Engine::setParams(const Params& p) {
    ap_.anonymize.store(p.anonymize, std::memory_order_relaxed);
    ap_.pitchSemitones.store(p.pitchSemitones, std::memory_order_relaxed);
    ap_.formantPercent.store(p.formantPercent, std::memory_order_relaxed);
    ap_.intonation.store(p.intonation, std::memory_order_relaxed);
    ap_.tiltDb.store(p.tiltDb, std::memory_order_relaxed);
    ap_.clarity.store(p.clarity, std::memory_order_relaxed);
    ap_.noiseSuppression.store(p.noiseSuppression, std::memory_order_relaxed);
    ap_.outputGainDb.store(p.outputGainDb, std::memory_order_relaxed);
    ap_.agc.store(p.agc, std::memory_order_relaxed);
    ap_.reshapeDb.store(p.spectralReshapeDb, std::memory_order_relaxed);
    ap_.driftSt.store(p.pitchDriftSemitones, std::memory_order_relaxed);
    ap_.fjitterPct.store(p.formantJitterPercent, std::memory_order_relaxed);
    ap_.flatten.store(p.dynamicsFlatten, std::memory_order_relaxed);
    ap_.seed.store(p.variationSeed, std::memory_order_relaxed);
    ap_.autoHint.store(p.autoDirectionHint > 0 ? 1 : p.autoDirectionHint < 0 ? -1 : 0, std::memory_order_relaxed);
    ap_.direction.store(static_cast<int>(p.direction), std::memory_order_release);
}

Params Engine::params() const {
    Params p;
    p.direction = static_cast<Direction>(ap_.direction.load(std::memory_order_acquire));
    p.anonymize = ap_.anonymize.load(std::memory_order_relaxed);
    p.pitchSemitones = ap_.pitchSemitones.load(std::memory_order_relaxed);
    p.formantPercent = ap_.formantPercent.load(std::memory_order_relaxed);
    p.intonation = ap_.intonation.load(std::memory_order_relaxed);
    p.tiltDb = ap_.tiltDb.load(std::memory_order_relaxed);
    p.clarity = ap_.clarity.load(std::memory_order_relaxed);
    p.noiseSuppression = ap_.noiseSuppression.load(std::memory_order_relaxed);
    p.outputGainDb = ap_.outputGainDb.load(std::memory_order_relaxed);
    p.agc = ap_.agc.load(std::memory_order_relaxed);
    p.autoDirectionHint = ap_.autoHint.load(std::memory_order_relaxed);
    p.spectralReshapeDb = ap_.reshapeDb.load(std::memory_order_relaxed);
    p.pitchDriftSemitones = ap_.driftSt.load(std::memory_order_relaxed);
    p.formantJitterPercent = ap_.fjitterPct.load(std::memory_order_relaxed);
    p.dynamicsFlatten = ap_.flatten.load(std::memory_order_relaxed);
    p.variationSeed = ap_.seed.load(std::memory_order_relaxed);
    return p;
}

Metrics Engine::metrics() const {
    Metrics m;
    m.inputRmsDb = am_.inRms.load(std::memory_order_relaxed);
    m.inputPeakDb = am_.inPeak.load(std::memory_order_relaxed);
    m.outputRmsDb = am_.outRms.load(std::memory_order_relaxed);
    m.outputPeakDb = am_.outPeak.load(std::memory_order_relaxed);
    m.f0In = am_.f0In.load(std::memory_order_relaxed);
    m.f0OutEstimate = am_.f0Out.load(std::memory_order_relaxed);
    m.meanF0 = am_.meanF0.load(std::memory_order_relaxed);
    m.voiced = am_.voiced.load(std::memory_order_relaxed);
    m.vad = am_.vad.load(std::memory_order_relaxed);
    m.pitchRatio = am_.pitchRatio.load(std::memory_order_relaxed);
    m.formantRatio = am_.formantRatio.load(std::memory_order_relaxed);
    m.direction = am_.direction.load(std::memory_order_relaxed);
    m.noiseFloorDb = am_.noiseFloor.load(std::memory_order_relaxed);
    m.limiterGainDb = am_.limiterGain.load(std::memory_order_relaxed);
    m.agcGainDb = am_.agcGain.load(std::memory_order_relaxed);
    m.lostFrames = am_.lost.load(std::memory_order_relaxed);
    m.sanitizedSamples = am_.sanitized.load(std::memory_order_relaxed);
    m.processedFrames = am_.processed.load(std::memory_order_relaxed);
    m.latencyMs = static_cast<float>(latencyMs());
    return m;
}

void Engine::process(const float* in, float* out, int n) {
    const float concealDecay = std::exp(-1.0f / (0.002f * static_cast<float>(fs_)));
    int done = 0;
    uint64_t lost = 0, sanitized = 0;
    while (done < n) {
        const int take = std::min(n - done, hop_ - inFill_);
        for (int i = 0; i < take; ++i) {
            float x;
            if (in == nullptr) {
                // Lost input: decay smoothly from the last real sample to silence.
                lastInput_ *= concealDecay;
                x = lastInput_;
                fadeInRemaining_ = fadeLen_;
                ++lost;
            } else {
                x = in[done + i];
                if (!std::isfinite(x)) {
                    x = 0.0f;
                    ++sanitized;
                } else if (x > 4.0f || x < -4.0f) {
                    x = clampf(x, -4.0f, 4.0f);
                    ++sanitized;
                }
                if (fadeInRemaining_ > 0) {
                    // Cross-fade from the concealment tail back to real input.
                    lastInput_ *= concealDecay;
                    const float r = 1.0f - static_cast<float>(fadeInRemaining_) / fadeLen_;
                    x = r * x + (1.0f - r) * lastInput_;
                    --fadeInRemaining_;
                }
                lastInput_ = x;
            }
            inFifo_[inFill_ + i] = x;
        }
        inFill_ += take;
        if (inFill_ == hop_) {
            processQuantum(inFifo_.data(), qOut_.data());
            inFill_ = 0;
            const int cap = static_cast<int>(outFifo_.size());
            for (int i = 0; i < hop_; ++i) outFifo_[(outRead_ + outFill_ + i) % cap] = qOut_[i];
            outFill_ += hop_;
        }
        const int cap = static_cast<int>(outFifo_.size());
        for (int i = 0; i < take; ++i) {
            out[done + i] = outFifo_[outRead_];
            outRead_ = (outRead_ + 1) % cap;
        }
        outFill_ -= take;
        done += take;
    }
    if (lost) am_.lost.fetch_add(lost, std::memory_order_relaxed);
    if (sanitized) am_.sanitized.fetch_add(sanitized, std::memory_order_relaxed);
    am_.processed.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
}

void Engine::updateEq(bool force) {
    if (!force && std::fabs(sTilt_ - eqTilt_) < 0.05f && std::fabs(sClarity_ - eqClarity_) < 0.01f) return;
    eqTilt_ = sTilt_;
    eqClarity_ = sClarity_;
    lowShelf_.setCoefs(Biquad::lowShelf(fs_, 250.0, -0.5 * eqTilt_));
    highShelf_.setCoefs(Biquad::highShelf(fs_, 3000.0, eqTilt_));
    presence_.setCoefs(Biquad::peaking(fs_, 3000.0, 0.9, 4.0 * eqClarity_));
    mud_.setCoefs(Biquad::peaking(fs_, 350.0, 1.0, -2.0 * eqClarity_));
}

void Engine::updateSmoothedParams(bool voicedNow, float f0Now, bool vadNow) {
    const Params p = params();
    const double dt = hop_ / fs_;
    if (voicedNow && vadNow && f0Now > 0.0f) {
        const double alpha = voicedSeconds_ < 1.0 ? dt / (voicedSeconds_ + dt) : dt / 2.0;
        logF0Mean_ += alpha * (std::log(static_cast<double>(f0Now)) - logF0Mean_);
        voicedSeconds_ += dt;
    }
    int sign;
    if (p.direction == Direction::Auto) {
        if (latchedDirection_ == 0 && voicedSeconds_ >= kDirectionLatchSeconds) {
            latchedDirection_ = std::exp(logF0Mean_) < kAutoF0Split ? 1 : -1;
        }
        sign = latchedDirection_ != 0 ? latchedDirection_ : (p.autoDirectionHint < 0 ? -1 : 1);
    } else {
        sign = p.direction == Direction::Down ? -1 : 1;
    }

    const float targetPitch = sign * clampf(p.pitchSemitones, -7.0f, 7.0f) / 12.0f;
    float r = 1.0f + clampf(p.formantPercent, -20.0f, 25.0f) / 100.0f;
    if (sign < 0) r = 1.0f / r;
    const float targetFormant =
        std::log2(clampf(r, PsolaShifter::kMinFormantRatio, PsolaShifter::kMaxFormantRatio));
    // Slow random processes (controlled prosody / micro-variation): a new random
    // target every 0.4-0.8 s, approached with a ~0.25 s one-pole.
    if (p.variationSeed != rngSeed_) {
        rngSeed_ = p.variationSeed;
        rng_ = p.variationSeed ? p.variationSeed * 2654435761u + 12345u : 12345u;
    }
    const int quantaPerSec = static_cast<int>(fs_ / hop_);
    if (--driftCountdown_ <= 0) {
        driftTarget_ = nextRandom();
        driftCountdown_ = quantaPerSec * 2 / 5 + static_cast<int>((nextRandom() + 1.0f) * quantaPerSec / 5);
    }
    if (--fjCountdown_ <= 0) {
        fjTarget_ = nextRandom();
        fjCountdown_ = quantaPerSec * 2 / 5 + static_cast<int>((nextRandom() + 1.0f) * quantaPerSec / 5);
    }
    const float slow = static_cast<float>(1.0 - std::exp(-dt / 0.25));
    drift_ += slow * (driftTarget_ - drift_);
    fj_ += slow * (fjTarget_ - fj_);
    const float driftLog2 = drift_ * clampf(p.pitchDriftSemitones, 0.0f, 3.0f) / 12.0f;
    const float fjLog2 = std::log2(1.0f + fj_ * clampf(p.formantJitterPercent, 0.0f, 10.0f) / 100.0f);

    const float c = smoothCoef_;
    sPitchLog2_ += c * (targetPitch + driftLog2 - sPitchLog2_);
    sFormantLog2_ += c * (targetFormant + fjLog2 - sFormantLog2_);
    sIntonation_ += c * (clampf(p.intonation, 0.5f, 1.5f) - sIntonation_);
    sTilt_ += c * (sign * clampf(p.tiltDb, -6.0f, 6.0f) - sTilt_);
    sClarity_ += c * (clampf(p.clarity, 0.0f, 1.0f) - sClarity_);
    am_.direction.store(p.direction == Direction::Auto ? latchedDirection_ : sign, std::memory_order_relaxed);
}

void Engine::processQuantum(const float* in, float* out) {
    const Params p = params();

    // Input meters + dry delay line (latency-matched bypass / capture reference).
    float inPeak = 0.0f;
    double inEnergy = 0.0;
    for (int i = 0; i < hop_; ++i) {
        const float x = in[i];
        inPeak = std::max(inPeak, std::fabs(x));
        inEnergy += static_cast<double>(x) * x;
        if (dryDelayLen_ > 0) {
            work2_[i] = dryDelay_[dryPos_];
            dryDelay_[dryPos_] = x;
            dryPos_ = (dryPos_ + 1) % dryDelayLen_;
        } else {
            work2_[i] = x;
        }
    }

    // High-pass + noise suppression (+ VAD).
    for (int i = 0; i < hop_; ++i) work_[i] = hpf_.process(in[i]);
    hpf_.flush();
    ns_.setAmount(clampf(p.noiseSuppression, 0.0f, 1.0f));
    updateReshape(clampf(p.spectralReshapeDb, 0.0f, 12.0f), p.variationSeed);
    ns_.processHop(work_.data(), work_.data());
    const bool vad = ns_.voiceActive();

    // Pitch analysis.
    tracker_.push(work_.data(), hop_);
    const bool voiced = tracker_.voiced();
    const float f0 = tracker_.f0();
    updateSmoothedParams(voiced, f0, vad);

    // Anonymization: pitch + formant + intonation.
    const float pitchRatio = std::exp2(sPitchLog2_);
    const float formantRatio = std::exp2(sFormantLog2_);
    const float f0Ref = voicedSeconds_ > 0.5 ? static_cast<float>(std::exp(logF0Mean_)) : 0.0f;
    psola_.process(work_.data(), qIn_.data(), hop_, tracker_, pitchRatio, formantRatio, sIntonation_, f0Ref);

    // Spectral colour.
    updateEq(false);
    double wetEnergy = 0.0;
    for (int i = 0; i < hop_; ++i) {
        float y = qIn_[i];
        y = lowShelf_.process(y);
        y = highShelf_.process(y);
        y = presence_.process(y);
        y = mud_.process(y);
        qIn_[i] = y;
        wetEnergy += static_cast<double>(y) * y;
    }
    lowShelf_.flush();
    highShelf_.flush();
    presence_.flush();
    mud_.flush();

    // Level: speech-gated AGC, residual gate in pauses, user gain.
    const double dt = hop_ / fs_;
    const float msq = static_cast<float>(wetEnergy / hop_);
    const float envCoef = static_cast<float>(1.0 - std::exp(-dt / 0.1));
    rmsEnv_ += envCoef * (msq - rmsEnv_);
    if (p.agc && vad && rmsEnv_ > 1e-9f) {
        const float desired = clampf(kAgcTargetDb - toDb(rmsEnv_), kAgcMinDb, kAgcMaxDb);
        const float tau = desired < agcGainDb_ ? 0.3f : 1.0f;
        agcGainDb_ += static_cast<float>(1.0 - std::exp(-dt / tau)) * (desired - agcGainDb_);
    } else if (!p.agc) {
        agcGainDb_ += static_cast<float>(1.0 - std::exp(-dt / 0.3)) * (0.0f - agcGainDb_);
    }
    silentQuanta_ = vad ? 0 : silentQuanta_ + 1;
    const bool longPause = silentQuanta_ * dt > 0.3;
    const float gateTarget =
        longPause ? std::pow(10.0f, -10.0f * clampf(p.noiseSuppression, 0.0f, 1.0f) / 20.0f) : 1.0f;
    const float gateStart = gateGain_;
    gateGain_ += static_cast<float>(1.0 - std::exp(-dt / 0.05)) * (gateTarget - gateGain_);
    const float outStart = sOutGain_;
    sOutGain_ += smoothCoef_ * (std::pow(10.0f, clampf(p.outputGainDb, -12.0f, 12.0f) / 20.0f) - sOutGain_);
    const float agcLin = std::pow(10.0f, agcGainDb_ / 20.0f);
    for (int i = 0; i < hop_; ++i) {
        const float t = static_cast<float>(i + 1) / hop_;
        const float g = (gateStart + t * (gateGain_ - gateStart)) * (outStart + t * (sOutGain_ - outStart));
        qIn_[i] *= g * agcLin;
    }
    // Energy-contour flattening: fast compressor (5 ms attack, 80 ms release)
    // above -30 dBFS, ratio 1 + 3 * amount, no look-ahead (no added latency).
    const float flatten = clampf(p.dynamicsFlatten, 0.0f, 1.0f);
    if (flatten > 0.0f) {
        const float att = static_cast<float>(1.0 - std::exp(-1.0 / (0.005 * fs_)));
        const float rel = static_cast<float>(1.0 - std::exp(-1.0 / (0.080 * fs_)));
        const float thr = 0.0316f;  // -30 dBFS
        const float slope = 1.0f - 1.0f / (1.0f + 3.0f * flatten);
        for (int i = 0; i < hop_; ++i) {
            const float a = std::fabs(qIn_[i]);
            dynEnv_ += (a > dynEnv_ ? att : rel) * (a - dynEnv_);
            if (dynEnv_ > thr) qIn_[i] *= std::pow(thr / dynEnv_, slope);
        }
    }

    limiter_.process(qIn_.data(), hop_);

    // Wet/dry cross-fade (A/B bypass) - 20 ms, click free.
    const float mixTarget = p.anonymize ? 1.0f : 0.0f;
    const float mixStep = static_cast<float>(1.0 / (0.02 * fs_));
    const float ceil = limiter_.ceiling();
    float outPeak = 0.0f;
    double outEnergy = 0.0;
    for (int i = 0; i < hop_; ++i) {
        if (mix_ < mixTarget) mix_ = std::min(mixTarget, mix_ + mixStep);
        if (mix_ > mixTarget) mix_ = std::max(mixTarget, mix_ - mixStep);
        const float dry = clampf(work2_[i], -ceil, ceil);
        const float y = mix_ * qIn_[i] + (1.0f - mix_) * dry;
        out[i] = y;
        outPeak = std::max(outPeak, std::fabs(y));
        outEnergy += static_cast<double>(y) * y;
    }

    // In-memory capture for test mode (dry is latency-aligned with wet).
    if (captureState_.load(std::memory_order_acquire) == 1) {
        int pos = capPos_.load(std::memory_order_relaxed);
        const int m = std::min(hop_, capLen_ - pos);
        for (int i = 0; i < m; ++i) {
            capDry_[pos + i] = work2_[i];
            capWet_[pos + i] = qIn_[i];
        }
        pos += m;
        capPos_.store(pos, std::memory_order_relaxed);
        if (pos >= capLen_) captureState_.store(2, std::memory_order_release);
    }

    // Metrics.
    am_.inRms.store(toDb(static_cast<float>(inEnergy / hop_)), std::memory_order_relaxed);
    am_.inPeak.store(20.0f * std::log10(inPeak + 1e-9f), std::memory_order_relaxed);
    am_.outRms.store(toDb(static_cast<float>(outEnergy / hop_)), std::memory_order_relaxed);
    am_.outPeak.store(20.0f * std::log10(outPeak + 1e-9f), std::memory_order_relaxed);
    am_.f0In.store(f0, std::memory_order_relaxed);
    am_.f0Out.store(voiced ? f0 * psola_.lastEffectivePitchRatio() : 0.0f, std::memory_order_relaxed);
    am_.meanF0.store(voicedSeconds_ > 0.0 ? static_cast<float>(std::exp(logF0Mean_)) : 0.0f,
                     std::memory_order_relaxed);
    am_.voiced.store(voiced, std::memory_order_relaxed);
    am_.vad.store(vad, std::memory_order_relaxed);
    am_.pitchRatio.store(pitchRatio, std::memory_order_relaxed);
    am_.formantRatio.store(formantRatio, std::memory_order_relaxed);
    am_.noiseFloor.store(ns_.noiseFloorDb(), std::memory_order_relaxed);
    am_.limiterGain.store(limiter_.gainReductionDb(), std::memory_order_relaxed);
    am_.agcGain.store(agcGainDb_, std::memory_order_relaxed);
}

bool Engine::startCapture(double seconds) {
    if (captureState_.load(std::memory_order_acquire) == 1) return false;
    const int len = static_cast<int>(std::max(0.1, std::min(seconds, 60.0)) * fs_);
    capDry_.assign(len, 0.0f);
    capWet_.assign(len, 0.0f);
    capLen_ = len;
    capPos_.store(0, std::memory_order_relaxed);
    captureState_.store(1, std::memory_order_release);
    return true;
}

float Engine::captureProgress() const {
    const int st = captureState_.load(std::memory_order_acquire);
    if (st == 2) return 1.0f;
    if (st == 0 || capLen_ == 0) return 0.0f;
    return static_cast<float>(capPos_.load(std::memory_order_relaxed)) / capLen_;
}

int Engine::readCapture(std::vector<float>& dry, std::vector<float>& wet) {
    if (captureState_.load(std::memory_order_acquire) != 2) return 0;
    dry = capDry_;
    wet = capWet_;
    captureState_.store(0, std::memory_order_release);
    return capLen_;
}

}  // namespace voiceanon
