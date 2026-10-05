#include "voiceanon/biquad.h"

#include <algorithm>

namespace voiceanon {
namespace {

Biquad normalize(double b0, double b1, double b2, double a0, double a1, double a2) {
    Biquad q;
    q.b0 = static_cast<float>(b0 / a0);
    q.b1 = static_cast<float>(b1 / a0);
    q.b2 = static_cast<float>(b2 / a0);
    q.a1 = static_cast<float>(a1 / a0);
    q.a2 = static_cast<float>(a2 / a0);
    return q;
}

double clampFreq(double fs, double f0) { return std::min(std::max(f0, 1.0), 0.49 * fs); }

}  // namespace

Biquad Biquad::lowpass(double fs, double f0, double q) {
    const double w = 2.0 * M_PI * clampFreq(fs, f0) / fs;
    const double c = std::cos(w), alpha = std::sin(w) / (2.0 * q);
    return normalize((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + alpha, -2 * c, 1 - alpha);
}

Biquad Biquad::highpass(double fs, double f0, double q) {
    const double w = 2.0 * M_PI * clampFreq(fs, f0) / fs;
    const double c = std::cos(w), alpha = std::sin(w) / (2.0 * q);
    return normalize((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + alpha, -2 * c, 1 - alpha);
}

Biquad Biquad::bandpass(double fs, double f0, double q) {
    const double w = 2.0 * M_PI * clampFreq(fs, f0) / fs;
    const double c = std::cos(w), alpha = std::sin(w) / (2.0 * q);
    return normalize(alpha, 0, -alpha, 1 + alpha, -2 * c, 1 - alpha);
}

Biquad Biquad::peaking(double fs, double f0, double q, double gainDb) {
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w = 2.0 * M_PI * clampFreq(fs, f0) / fs;
    const double c = std::cos(w), alpha = std::sin(w) / (2.0 * q);
    return normalize(1 + alpha * A, -2 * c, 1 - alpha * A, 1 + alpha / A, -2 * c, 1 - alpha / A);
}

Biquad Biquad::lowShelf(double fs, double f0, double gainDb) {
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w = 2.0 * M_PI * clampFreq(fs, f0) / fs;
    const double c = std::cos(w), s = std::sin(w);
    const double alpha = s / 2.0 * std::sqrt(2.0);  // shelf slope S = 1
    const double sq = 2.0 * std::sqrt(A) * alpha;
    return normalize(A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c),
                     A * ((A + 1) - (A - 1) * c - sq), (A + 1) + (A - 1) * c + sq,
                     -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq);
}

Biquad Biquad::highShelf(double fs, double f0, double gainDb) {
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w = 2.0 * M_PI * clampFreq(fs, f0) / fs;
    const double c = std::cos(w), s = std::sin(w);
    const double alpha = s / 2.0 * std::sqrt(2.0);
    const double sq = 2.0 * std::sqrt(A) * alpha;
    return normalize(A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c),
                     A * ((A + 1) + (A - 1) * c - sq), (A + 1) - (A - 1) * c + sq,
                     2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq);
}

}  // namespace voiceanon
