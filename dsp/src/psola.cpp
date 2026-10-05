#include "voiceanon/psola.h"

#include <algorithm>
#include <cmath>

#include "voiceanon/fft.h"

namespace voiceanon {
namespace {
constexpr int kMarkRing = 256;
}

PsolaShifter::PsolaShifter(double sampleRate) : fs_(sampleRate) {
    hLeftMax_ = static_cast<int>(std::lround(0.012 * fs_));
    hRightInMax_ = static_cast<int>(std::lround(0.008 * fs_));
    tu_ = static_cast<int>(std::lround(0.005 * fs_));
    latency_ = hLeftMax_ + hRightInMax_ + 2;
    stride_ = std::max(1, static_cast<int>(fs_ / 12000.0));
    const int size = nextPow2(static_cast<int>(fs_ * 0.25));
    in_.assign(size, 0.0f);
    acc_.assign(size, 0.0f);
    wsum_.assign(size, 0.0f);
    mask_ = size - 1;
    marks_.resize(kMarkRing);
}

void PsolaShifter::reset() {
    std::fill(in_.begin(), in_.end(), 0.0f);
    std::fill(acc_.begin(), acc_.end(), 0.0f);
    std::fill(wsum_.begin(), wsum_.end(), 0.0f);
    written_ = 0;
    markCount_ = 0;
    kappa_ = 0.0;
    lastPitch_ = 1.0f;
}

float PsolaShifter::sampleAt(int64_t i) const {
    if (i < 0 || i >= written_ || i < written_ - static_cast<int64_t>(in_.size())) return 0.0f;
    return in_[i & mask_];
}

float PsolaShifter::interp(double pos) const {
    const double fl = std::floor(pos);
    const int64_t i = static_cast<int64_t>(fl);
    const float t = static_cast<float>(pos - fl);
    const float x0 = sampleAt(i - 1), x1 = sampleAt(i), x2 = sampleAt(i + 1), x3 = sampleAt(i + 2);
    // Catmull-Rom cubic
    return x1 + 0.5f * t * (x2 - x0 + t * (2.0f * x0 - 5.0f * x1 + 4.0f * x2 - x3 + t * (3.0f * (x1 - x2) + x3 - x0)));
}

int64_t PsolaShifter::refine(int64_t prev, int64_t expected, double period, int64_t limit) const {
    const int r = std::max(1, static_cast<int>(period / 4));
    const int kLo = -static_cast<int>(period / 2), kHi = static_cast<int>(period / 4);
    const int64_t lower = prev + static_cast<int64_t>(period / 2);
    auto score = [&](int64_t c) {
        double xy = 0.0, xx = 0.0, yy = 0.0;
        for (int k = kLo; k <= kHi; k += stride_) {
            const float a = sampleAt(prev + k), b = sampleAt(c + k);
            xy += static_cast<double>(a) * b;
            xx += static_cast<double>(a) * a;
            yy += static_cast<double>(b) * b;
        }
        return xy / std::sqrt(xx * yy + 1e-18);
    };
    int64_t best = std::min(expected, limit);
    double bestScore = -2.0;
    for (int d = -r; d <= r; d += stride_) {
        const int64_t c = expected + d;
        if (c <= lower || c > limit) continue;
        const double s = score(c);
        if (s > bestScore) {
            bestScore = s;
            best = c;
        }
    }
    if (stride_ > 1) {  // fine search around the coarse optimum
        const int64_t coarse = best;
        for (int d = -stride_ + 1; d < stride_; ++d) {
            const int64_t c = coarse + d;
            if (d == 0 || c <= lower || c > limit) continue;
            const double s = score(c);
            if (s > bestScore) {
                bestScore = s;
                best = c;
            }
        }
    }
    return best;
}

int64_t PsolaShifter::snapToPeak(int64_t expected, double period, int64_t lowerBound, int64_t limit) const {
    const int r = std::max(1, static_cast<int>(period / 4));
    int64_t best = std::min(expected, limit);
    float bestVal = -1.0f;
    for (int64_t c = std::max(expected - r, lowerBound + 1); c <= std::min(expected + r, limit); ++c) {
        const float v = std::fabs(sampleAt(c));
        if (v > bestVal) {
            bestVal = v;
            best = c;
        }
    }
    return best;
}

void PsolaShifter::generateMarks(int64_t frontier, const PitchTracker& tracker) {
    for (;;) {
        const bool voiced = tracker.voiced();
        double period = static_cast<double>(tu_);
        if (voiced) {
            period = std::min(std::max(tracker.periodSamples(), fs_ / PitchTracker::kMaxF0),
                              fs_ / PitchTracker::kMinF0);
        }
        const bool haveLast = markCount_ > 0;
        Mark last{0, 0.0, false};
        if (haveLast) last = marks_[(markCount_ - 1) % kMarkRing];
        const int64_t expected = haveLast ? last.pos + static_cast<int64_t>(std::lround(period)) : 0;
        if (expected > frontier) break;

        int64_t pos = expected;
        if (voiced && haveLast) {
            if (last.voiced) {
                pos = refine(last.pos, expected, period, frontier);
            } else {
                pos = snapToPeak(expected, period, last.pos, frontier);
            }
        }
        if (haveLast && pos <= last.pos) pos = last.pos + 1;
        marks_[markCount_ % kMarkRing] = Mark{pos, period, voiced};
        ++markCount_;
    }
}

const PsolaShifter::Mark& PsolaShifter::markAt(int64_t index) const {
    const int64_t oldest = std::max<int64_t>(0, markCount_ - kMarkRing);
    index = std::min(std::max(index, oldest), markCount_ - 1);
    return marks_[index % kMarkRing];
}

double PsolaShifter::analysisTime(double kappa) const {
    const int64_t last = markCount_ - 1;
    const int64_t k = std::min(static_cast<int64_t>(std::floor(kappa)), last);
    const double frac = kappa - static_cast<double>(k);  // may exceed 1 beyond the newest mark
    const Mark& m = markAt(k);
    if (k < last) {
        const Mark& n = markAt(k + 1);
        return static_cast<double>(m.pos) + frac * static_cast<double>(n.pos - m.pos);
    }
    return static_cast<double>(m.pos) + frac * (m.voiced ? m.period : static_cast<double>(tu_));
}

void PsolaShifter::process(const float* in, float* out, int n, const PitchTracker& tracker, float pitchRatio,
                           float formantRatio, float intonation, float f0Ref) {
    for (int i = 0; i < n; ++i) in_[(written_ + i) & mask_] = in[i];
    written_ += n;

    const double f = std::min(std::max(static_cast<double>(formantRatio), static_cast<double>(kMinFormantRatio)),
                              static_cast<double>(kMaxFormantRatio));
    // Grains centred at or before the frontier have all the input they need.
    const int64_t frontier = written_ - 1 - hRightInMax_ - 2;
    if (frontier >= 0) generateMarks(frontier, tracker);

    while (markCount_ > 0) {
        const int64_t oldest = std::max<int64_t>(0, markCount_ - kMarkRing);
        if (kappa_ < static_cast<double>(oldest)) kappa_ = static_cast<double>(oldest);
        const double s = analysisTime(kappa_);
        if (s > static_cast<double>(frontier)) break;
        const Mark& m = markAt(static_cast<int64_t>(std::llround(kappa_)));

        double advance, span;
        if (m.voiced) {
            const double ta = m.period;
            double pe = pitchRatio;
            if (f0Ref > 0.0f && intonation != 1.0f) {
                const double f0 = fs_ / ta;
                pe *= std::pow(f0 / f0Ref, static_cast<double>(intonation) - 1.0);
            }
            pe = std::min(std::max(pe, 0.5), 2.0);
            lastPitch_ = static_cast<float>(pe);
            advance = 1.0 / pe;
            // >= one output period so neighbouring grains always overlap.
            span = std::max(ta / f, ta / pe);
        } else {
            advance = 1.0;
            span = tu_;
        }
        const double hl = std::min(std::max(span, 8.0), static_cast<double>(hLeftMax_));
        const double hr = std::min(std::max(span, 8.0), std::floor(hRightInMax_ / f));

        // Render the grain centred at (fractional) synthesis time s.
        const int64_t t0 = static_cast<int64_t>(std::ceil(s - hl + 1e-9));
        const int64_t t1 = static_cast<int64_t>(std::floor(s + hr - 1e-9));
        for (int64_t t = t0; t <= t1; ++t) {
            const double off = static_cast<double>(t) - s;
            const double ph = off < 0.0 ? off / hl : off / hr;
            const float w = 0.5f + 0.5f * static_cast<float>(std::cos(M_PI * ph));
            const float v = interp(static_cast<double>(m.pos) + off * f);
            const int64_t idx = t & mask_;
            acc_[idx] += w * v;
            wsum_[idx] += w;
        }
        kappa_ += advance;
    }

    const int64_t start = written_ - n - latency_;
    for (int i = 0; i < n; ++i) {
        const int64_t e = start + i;
        if (e < 0) {
            out[i] = 0.0f;
            continue;
        }
        const int64_t idx = e & mask_;
        const float w = wsum_[idx];
        out[i] = w > 0.0f ? acc_[idx] / std::max(w, 0.5f) : 0.0f;
        acc_[idx] = 0.0f;
        wsum_[idx] = 0.0f;
    }
}

}  // namespace voiceanon
