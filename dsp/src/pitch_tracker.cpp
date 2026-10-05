#include "voiceanon/pitch_tracker.h"

#include <algorithm>
#include <cmath>

#include "voiceanon/fft.h"

namespace voiceanon {

PitchTracker::PitchTracker(double sampleRate) : fs_(sampleRate) {
    dec_ = std::max(1, static_cast<int>(sampleRate / 11000.0));
    fsd_ = fs_ / dec_;
    const double cutoff = std::min(1100.0, 0.45 * fsd_);
    lp1_ = Biquad::lowpass(fs_, cutoff, 0.5412);  // 4th-order Butterworth
    lp2_ = Biquad::lowpass(fs_, cutoff, 1.3066);
    tauMin_ = static_cast<int>(std::floor(fsd_ / kMaxF0));
    tauMax_ = static_cast<int>(std::ceil(fsd_ / kMinF0));
    window_ = tauMax_;
    hop_ = std::max(1, static_cast<int>(std::lround(fsd_ * 0.005)));
    ring_.assign(nextPow2(2 * (window_ + tauMax_) + 8), 0.0f);
    mask_ = static_cast<int>(ring_.size()) - 1;
    buf_.assign(window_ + tauMax_ + 2, 0.0f);
    diff_.assign(tauMax_ + 2, 0.0f);
}

void PitchTracker::reset() {
    lp1_.reset();
    lp2_.reset();
    std::fill(ring_.begin(), ring_.end(), 0.0f);
    written_ = 0;
    decPhase_ = 0;
    sinceLast_ = 0;
    voiced_ = false;
    period_ = 0.0;
    confidence_ = 0.0f;
}

void PitchTracker::push(const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        const float y = lp2_.process(lp1_.process(x[i]));
        if (++decPhase_ >= dec_) {
            decPhase_ = 0;
            ring_[written_ & mask_] = y;
            ++written_;
            if (++sinceLast_ >= hop_) {
                sinceLast_ = 0;
                analyze();
            }
        }
    }
    lp1_.flush();
    lp2_.flush();
}

void PitchTracker::analyze() {
    const int len = window_ + tauMax_;
    if (written_ < len) return;
    double energy = 0.0;
    for (int i = 0; i < len; ++i) {
        const float v = ring_[(written_ - len + i) & mask_];
        buf_[i] = v;
        energy += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(energy / len);
    if (rms < 3e-4) {  // ~ -70 dBFS after low-pass: treat as silence
        voiced_ = false;
        confidence_ = 0.0f;
        return;
    }

    // YIN difference function + cumulative mean normalisation.
    diff_[0] = 1.0f;
    double running = 0.0;
    for (int tau = 1; tau <= tauMax_; ++tau) {
        double d = 0.0;
        for (int j = 0; j < window_; ++j) {
            const float delta = buf_[j] - buf_[j + tau];
            d += static_cast<double>(delta) * delta;
        }
        running += d;
        diff_[tau] = running > 0.0 ? static_cast<float>(d * tau / running) : 1.0f;
    }

    const float threshold = 0.15f;
    int best = -1;
    for (int tau = std::max(tauMin_, 2); tau < tauMax_; ++tau) {
        if (diff_[tau] < threshold) {
            while (tau + 1 < tauMax_ && diff_[tau + 1] < diff_[tau]) ++tau;
            best = tau;
            break;
        }
    }
    if (best < 0) {  // fall back to global minimum
        float m = 1e9f;
        for (int tau = std::max(tauMin_, 2); tau < tauMax_; ++tau) {
            if (diff_[tau] < m) {
                m = diff_[tau];
                best = tau;
            }
        }
    }
    // Continuity: if the previous frame was voiced and a comparably deep dip
    // exists near the previous lag, prefer it (suppresses octave jumps).
    if (voiced_ && period_ > 0.0) {
        const int prevTau = static_cast<int>(std::lround(period_ / dec_));
        if (std::abs(best - prevTau) > prevTau / 8) {
            const int lo = std::max(std::max(tauMin_, 2), prevTau - prevTau / 10);
            const int hi = std::min(tauMax_ - 1, prevTau + prevTau / 10);
            int cand = -1;
            for (int tau = lo; tau <= hi; ++tau) {
                if (diff_[tau] <= diff_[tau - 1] && diff_[tau] <= diff_[tau + 1] &&
                    (cand < 0 || diff_[tau] < diff_[cand]))
                    cand = tau;
            }
            if (cand > 0 && diff_[cand] < 0.3f && diff_[cand] < diff_[best] + 0.1f) best = cand;
        }
    }
    const float dip = diff_[best];
    const float voicingLimit = voiced_ ? 0.35f : 0.25f;
    if (best <= 0 || dip > voicingLimit) {
        voiced_ = false;
        confidence_ = std::max(0.0f, 1.0f - dip);
        return;
    }
    // Parabolic interpolation around the dip.
    double refined = best;
    if (best > 1 && best < tauMax_) {
        const double a = diff_[best - 1], b = diff_[best], c = diff_[best + 1];
        const double den = a - 2.0 * b + c;
        if (std::fabs(den) > 1e-9) refined = best + 0.5 * (a - c) / den;
    }
    voiced_ = true;
    period_ = refined * dec_;
    confidence_ = 1.0f - dip;
}

}  // namespace voiceanon
