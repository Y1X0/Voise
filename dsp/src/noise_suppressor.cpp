#include "voiceanon/noise_suppressor.h"

#include <algorithm>
#include <cmath>

namespace voiceanon {
namespace {
constexpr float kEps = 1e-12f;
}

NoiseSuppressor::NoiseSuppressor(double sampleRate)
    : fs_(sampleRate),
      n_(nextPow2(static_cast<int>(sampleRate * 0.0095))),
      hop_(n_ / 2),
      bins_(n_ / 2 + 1),
      fft_(n_) {
    window_.resize(n_);
    for (int i = 0; i < n_; ++i) {  // periodic sqrt-Hann
        window_[i] = static_cast<float>(std::sqrt(0.5 - 0.5 * std::cos(2.0 * M_PI * i / n_)));
    }
    frame_.assign(n_, 0.0f);
    ola_.assign(n_, 0.0f);
    tmp_.assign(n_, 0.0f);
    re_.assign(bins_, 0.0f);
    im_.assign(bins_, 0.0f);
    smooth_.assign(bins_, 0.0f);
    minimum_.assign(bins_, 0.0f);
    noise_.assign(bins_, 0.0f);
    gain_.assign(bins_, 1.0f);
    prevGain_.assign(bins_, 1.0f);
    prevPost_.assign(bins_, 1.0f);
    gainSmoothed_.assign(bins_, 1.0f);
    const double binHz = fs_ / n_;
    binLo_ = std::max(1, static_cast<int>(300.0 / binHz));
    binHi_ = std::min(bins_ - 1, static_cast<int>(std::min(3000.0, 0.45 * fs_) / binHz));
    binHiHigh_ = std::min(bins_ - 1, static_cast<int>(std::min(7500.0, 0.47 * fs_) / binHz));
    const double framesPerSec = fs_ / hop_;
    hangFrames_ = static_cast<int>(0.25 * framesPerSec);
    subLen_ = std::max(1, static_cast<int>(1.5 * framesPerSec / kSubWindows));
    subMin_.assign(static_cast<size_t>(bins_) * kSubWindows, 0.0f);
    actMin_.assign(bins_, 0.0f);
}

void NoiseSuppressor::setAmount(float amount) { amount_ = std::min(std::max(amount, 0.0f), 1.0f); }

void NoiseSuppressor::reset() {
    std::fill(frame_.begin(), frame_.end(), 0.0f);
    std::fill(ola_.begin(), ola_.end(), 0.0f);
    std::fill(gain_.begin(), gain_.end(), 1.0f);
    std::fill(prevGain_.begin(), prevGain_.end(), 1.0f);
    std::fill(prevPost_.begin(), prevPost_.end(), 1.0f);
    std::fill(gainSmoothed_.begin(), gainSmoothed_.end(), 1.0f);
    frames_ = 0;
    subCount_ = subFilled_ = subIndex_ = 0;
    vad_ = false;
    hang_ = 0;
}

void NoiseSuppressor::processHop(const float* in, float* out) {
    // Slide analysis frame and append the new hop.
    std::copy(frame_.begin() + hop_, frame_.end(), frame_.begin());
    double hopEnergy = 0.0;
    for (int i = 0; i < hop_; ++i) {
        frame_[n_ - hop_ + i] = in[i];
        hopEnergy += static_cast<double>(in[i]) * in[i];
    }
    const float hopRmsDb = 10.0f * std::log10(static_cast<float>(hopEnergy / hop_) + kEps);

    for (int i = 0; i < n_; ++i) tmp_[i] = frame_[i] * window_[i];
    fft_.forwardReal(tmp_.data(), re_.data(), im_.data());

    // --- Noise PSD tracking -------------------------------------------------
    const bool first = frames_ == 0;
    const bool startup = frames_ < 30;  // ~160 ms
    double sigSum = 0.0, noiseSum = 0.0, sigHigh = 0.0, noiseHigh = 0.0;
    const bool closeSubWindow = ++subCount_ >= subLen_;
    for (int k = 0; k < bins_; ++k) {
        const float p = re_[k] * re_[k] + im_[k] * im_[k];
        float* stored = &subMin_[static_cast<size_t>(k) * kSubWindows];
        if (first) {
            smooth_[k] = minimum_[k] = noise_[k] = actMin_[k] = p;
            for (int u = 0; u < kSubWindows; ++u) stored[u] = p;
        } else {
            smooth_[k] = 0.7f * smooth_[k] + 0.3f * p;
            actMin_[k] = std::min(actMin_[k], smooth_[k]);
            // Minimum statistics: min over the stored sub-window minima + current one.
            float m = actMin_[k];
            const int filled = std::max(1, subFilled_);
            for (int u = 0; u < filled; ++u) m = std::min(m, stored[u]);
            minimum_[k] = m;
            const bool noiseLike = smooth_[k] < 4.0f * minimum_[k];
            if (startup) {
                noise_[k] = 0.8f * noise_[k] + 0.2f * smooth_[k];
            } else if (noiseLike) {
                noise_[k] = 0.9f * noise_[k] + 0.1f * smooth_[k];
            }
            noise_[k] = std::max(noise_[k], minimum_[k]);
            if (closeSubWindow) {
                stored[subIndex_] = actMin_[k];
                actMin_[k] = smooth_[k];
            }
        }
        if (k >= binLo_ && k <= binHi_) {
            sigSum += smooth_[k];
            noiseSum += noise_[k];
        } else if (k > binHi_ && k <= binHiHigh_) {
            sigHigh += smooth_[k];
            noiseHigh += noise_[k];
        }
    }
    if (closeSubWindow) {
        subCount_ = 0;
        subIndex_ = (subIndex_ + 1) % kSubWindows;
        subFilled_ = std::min(kSubWindows, subFilled_ + 1);
    }
    ++frames_;

    // --- VAD -----------------------------------------------------------------
    // Speech if either the voiced band or the fricative band rises above noise.
    snrDb_ = 10.0f * std::log10(static_cast<float>((sigSum + kEps) / (noiseSum + kEps)));
    if (binHiHigh_ > binHi_) {
        const float hi = 10.0f * std::log10(static_cast<float>((sigHigh + kEps) / (noiseHigh + kEps)));
        snrDb_ = std::max(snrDb_, hi);
    }
    const float thr = vad_ ? 2.0f : 4.0f;
    const bool frameVoice = snrDb_ > thr && hopRmsDb > -70.0f;
    if (frameVoice) {
        hang_ = hangFrames_;
    } else if (hang_ > 0) {
        --hang_;
    }
    vad_ = hang_ > 0;
    // Noise level meter: for sqrt-Hann frames E|X_k|^2 = sigma^2 * n / 2.
    noiseDb_ = 10.0f * std::log10(static_cast<float>(noiseSum / (binHi_ - binLo_ + 1) / (0.5 * n_)) + kEps);

    // --- Gain ----------------------------------------------------------------
    const float floorGain = std::pow(10.0f, -20.0f * amount_ / 20.0f);
    if (amount_ <= 0.0f) {
        std::fill(gainSmoothed_.begin(), gainSmoothed_.end(), 1.0f);
    } else {
        for (int k = 0; k < bins_; ++k) {
            const float p = re_[k] * re_[k] + im_[k] * im_[k];
            // Mild over-subtraction that grows with the amount control.
            const float post = p / ((1.0f + amount_) * noise_[k] + kEps);
            // Decision-directed a-priori SNR (beta tuned for a ~5 ms hop).
            const float prio = 0.92f * prevGain_[k] * prevGain_[k] * prevPost_[k] +
                               0.08f * std::max(post - 1.0f, 0.0f);
            float g = prio / (1.0f + prio);
            g = std::max(g, floorGain);
            gain_[k] = g;
            prevGain_[k] = g;
            prevPost_[k] = std::min(post, 1e6f);
        }
        // Smooth across frequency to suppress musical noise.
        for (int k = 0; k < bins_; ++k) {
            const float l = gain_[k > 0 ? k - 1 : k];
            const float r = gain_[k < bins_ - 1 ? k + 1 : k];
            gainSmoothed_[k] = 0.25f * l + 0.5f * gain_[k] + 0.25f * r;
        }
    }
    for (int k = 0; k < bins_; ++k) {
        re_[k] *= gainSmoothed_[k];
        im_[k] *= gainSmoothed_[k];
    }

    fft_.inverseReal(re_.data(), im_.data(), tmp_.data());
    for (int i = 0; i < n_; ++i) ola_[i] += tmp_[i] * window_[i];
    for (int i = 0; i < hop_; ++i) out[i] = ola_[i];
    std::copy(ola_.begin() + hop_, ola_.end(), ola_.begin());
    std::fill(ola_.end() - hop_, ola_.end(), 0.0f);
}

}  // namespace voiceanon
