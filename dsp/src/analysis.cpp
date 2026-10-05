#include "voiceanon/analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "voiceanon/biquad.h"
#include "voiceanon/fft.h"

namespace voiceanon {
namespace analysis {
namespace {

// Low-pass + decimate to an analysis rate close to `target` (never upsamples).
std::vector<float> decimate(const float* x, int n, double fs, double target, double cutoff, double* fsOut) {
    const int d = std::max(1, static_cast<int>(fs / target));
    *fsOut = fs / d;
    Biquad a = Biquad::lowpass(fs, std::min(cutoff, 0.45 * fs / d), 0.5412);
    Biquad b = Biquad::lowpass(fs, std::min(cutoff, 0.45 * fs / d), 1.3066);
    std::vector<float> y;
    y.reserve(n / d + 1);
    for (int i = 0; i < n; ++i) {
        const float v = b.process(a.process(x[i]));
        if (i % d == 0) y.push_back(v);
    }
    return y;
}

double frameEnergy(const float* x, int len) {
    double e = 0.0;
    for (int i = 0; i < len; ++i) e += static_cast<double>(x[i]) * x[i];
    return e / std::max(1, len);
}

// Levinson-Durbin. Returns prediction error; a[0] = 1.
double levinson(const std::vector<double>& r, int order, std::vector<double>& a) {
    a.assign(order + 1, 0.0);
    a[0] = 1.0;
    double err = r[0];
    if (err <= 0.0) return 0.0;
    std::vector<double> tmp(order + 1);
    for (int i = 1; i <= order; ++i) {
        double acc = r[i];
        for (int j = 1; j < i; ++j) acc += a[j] * r[i - j];
        const double k = -acc / err;
        tmp = a;
        for (int j = 1; j < i; ++j) a[j] = tmp[j] + k * tmp[i - j];
        a[i] = k;
        err *= (1.0 - k * k);
        if (err <= 1e-15) break;
    }
    return err;
}

double interpVec(const std::vector<float>& v, double idx) {
    if (idx <= 0) return v.front();
    const double last = static_cast<double>(v.size() - 1);
    if (idx >= last) return v.back();
    const int i = static_cast<int>(idx);
    const double t = idx - i;
    return v[i] * (1.0 - t) + v[i + 1] * t;
}

double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 3) return 0.0;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / std::sqrt(saa * sbb + 1e-30);
}

}  // namespace

double rmsDb(const float* x, int n) { return 10.0 * std::log10(frameEnergy(x, n) + 1e-20); }

std::vector<float> trackPitch(const float* x, int n, double fs) {
    // Independent of the engine's YIN tracker: normalised cross-correlation (NCCF)
    // on an ~8 kHz low-passed copy, 10 ms hop, 20 ms correlation window.
    double fsA = fs;
    const std::vector<float> y = decimate(x, n, fs, 8000.0, 1000.0, &fsA);
    const int hop = static_cast<int>(std::lround(0.010 * fsA));
    const int win = static_cast<int>(std::lround(0.020 * fsA));
    const int tauMin = static_cast<int>(std::floor(fsA / 500.0));
    const int tauMax = static_cast<int>(std::ceil(fsA / 65.0));
    const int frames = (n > 0) ? static_cast<int>(y.size()) / hop : 0;

    double maxE = 0.0;
    std::vector<double> energies(frames, 0.0);
    for (int f = 0; f < frames; ++f) {
        const int s = f * hop;
        if (s + win > static_cast<int>(y.size())) break;
        energies[f] = frameEnergy(&y[s], win);
        maxE = std::max(maxE, energies[f]);
    }
    std::vector<float> f0(frames, 0.0f);
    std::vector<double> nccf(tauMax + 2, 0.0);
    for (int f = 0; f < frames; ++f) {
        const int s = f * hop;
        if (s + win + tauMax + 1 >= static_cast<int>(y.size())) break;
        if (energies[f] < maxE * 1e-4 || energies[f] < 1e-9) continue;  // 40 dB below loudest / silence
        double e0 = 0.0;
        for (int j = 0; j < win; ++j) e0 += static_cast<double>(y[s + j]) * y[s + j];
        double best = 0.0;
        for (int tau = tauMin; tau <= tauMax + 1; ++tau) {
            double xy = 0.0, e1 = 0.0;
            for (int j = 0; j < win; ++j) {
                xy += static_cast<double>(y[s + j]) * y[s + j + tau];
                e1 += static_cast<double>(y[s + j + tau]) * y[s + j + tau];
            }
            nccf[tau] = xy / std::sqrt(e0 * e1 + 1e-30);
            best = std::max(best, nccf[tau]);
        }
        if (best < 0.6) continue;
        // Smallest-lag local peak within 90 % of the best one (avoids octave-down errors).
        int pick = -1;
        for (int tau = tauMin + 1; tau <= tauMax; ++tau) {
            if (nccf[tau] >= 0.9 * best && nccf[tau] >= nccf[tau - 1] && nccf[tau] >= nccf[tau + 1]) {
                pick = tau;
                break;
            }
        }
        if (pick < 0) continue;
        double refined = pick;
        const double a = nccf[pick - 1], b = nccf[pick], c = nccf[pick + 1];
        const double den = a - 2.0 * b + c;
        if (std::fabs(den) > 1e-12) refined = pick + 0.5 * (a - c) / den;
        f0[f] = static_cast<float>(fsA / refined);
    }
    return f0;
}

float medianVoicedF0(const std::vector<float>& f0) {
    std::vector<float> v;
    for (float x : f0)
        if (x > 0) v.push_back(x);
    if (v.empty()) return 0.0f;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

std::vector<float> averageEnvelopeDb(const float* x, int n, double fs, double maxHz, int points) {
    double fsA = fs;
    const std::vector<float> y = decimate(x, n, fs, 16000.0, 7600.0, &fsA);
    const int win = static_cast<int>(std::lround(0.025 * fsA));
    const int hop = static_cast<int>(std::lround(0.010 * fsA));
    const int order = std::min(22, 2 + static_cast<int>(fsA / 1000.0));
    std::vector<double> e;
    for (int s = 0; s + win <= static_cast<int>(y.size()); s += hop) e.push_back(frameEnergy(&y[s], win));
    const double maxE = e.empty() ? 0.0 : *std::max_element(e.begin(), e.end());
    std::vector<double> sum(points, 0.0);
    int count = 0;
    std::vector<double> frame(win), r(order + 1), a;
    for (size_t fi = 0; fi < e.size(); ++fi) {
        if (e[fi] < maxE * 1e-2 || e[fi] < 1e-10) continue;  // speech-dominant: within 20 dB of the loudest frame
        const int s = static_cast<int>(fi) * hop;
        for (int i = 0; i < win; ++i) {
            const double pre = y[s + i] - (s + i > 0 ? 0.9 * y[s + i - 1] : 0.0);
            frame[i] = pre * (0.5 - 0.5 * std::cos(2.0 * M_PI * i / (win - 1)));
        }
        for (int k = 0; k <= order; ++k) {
            double acc = 0.0;
            for (int i = k; i < win; ++i) acc += frame[i] * frame[i - k];
            r[k] = acc;
        }
        r[0] *= 1.0001;  // white-noise correction
        const double err = levinson(r, order, a);
        if (err <= 0.0) continue;
        for (int p = 0; p < points; ++p) {
            const double w = 2.0 * M_PI * (maxHz * p / (points - 1)) / fsA;
            double re = 0.0, im = 0.0;
            for (int k = 0; k <= order; ++k) {
                re += a[k] * std::cos(w * k);
                im -= a[k] * std::sin(w * k);
            }
            sum[p] += 10.0 * std::log10(err / (re * re + im * im + 1e-30) + 1e-30);
        }
        ++count;
    }
    std::vector<float> out(points, -120.0f);
    if (count > 0)
        for (int p = 0; p < points; ++p) out[p] = static_cast<float>(sum[p] / count);
    return out;
}

double estimateFormantRatio(const float* a, const float* b, int n, double fs) {
    const double maxHz = std::min(5000.0, 0.45 * std::min(fs, 16000.0));
    const int points = 500;
    const std::vector<float> ea = averageEnvelopeDb(a, n, fs, maxHz, points);
    const std::vector<float> eb = averageEnvelopeDb(b, n, fs, maxHz, points);
    const double lo = 300.0, hi = std::min(3800.0, maxHz * 0.70);
    double bestAlpha = 1.0, bestErr = 1e30;
    for (double alpha = 0.70; alpha <= 1.40001; alpha += 0.002) {
        // Residual after removing a best-fit line (gain + tilt) from the difference.
        std::vector<double> fx, dy;
        for (double f = lo; f <= hi; f += 10.0) {
            const double src = f / alpha;
            if (src > maxHz) break;
            const double vb = interpVec(eb, f / maxHz * (points - 1));
            const double va = interpVec(ea, src / maxHz * (points - 1));
            fx.push_back(f / 1000.0);
            dy.push_back(vb - va);
        }
        if (fx.size() < 50) continue;
        const double m = static_cast<double>(fx.size());
        const double sx = std::accumulate(fx.begin(), fx.end(), 0.0);
        const double sy = std::accumulate(dy.begin(), dy.end(), 0.0);
        double sxx = 0, sxy = 0;
        for (size_t i = 0; i < fx.size(); ++i) {
            sxx += fx[i] * fx[i];
            sxy += fx[i] * dy[i];
        }
        const double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx + 1e-30);
        const double icpt = (sy - slope * sx) / m;
        double err = 0.0;
        for (size_t i = 0; i < fx.size(); ++i) {
            const double res = dy[i] - (icpt + slope * fx[i]);
            err += res * res;
        }
        err /= m;
        if (err < bestErr) {
            bestErr = err;
            bestAlpha = alpha;
        }
    }
    // A minimum on the search boundary means the estimate is not trustworthy
    // (typically short, noisy or very quiet material).
    if (bestAlpha < 0.71 || bestAlpha > 1.39) return -1.0;
    return bestAlpha;
}

std::vector<float> meanMfcc(const float* x, int n, double fs) {
    double fsA = fs;
    const std::vector<float> y = decimate(x, n, fs, 16000.0, 7600.0, &fsA);
    const int win = static_cast<int>(std::lround(0.025 * fsA));
    const int hop = static_cast<int>(std::lround(0.010 * fsA));
    const int nfft = nextPow2(win);
    const int bands = 26, ceps = 13;
    Fft fft(nfft);
    std::vector<float> buf(nfft), re(nfft / 2 + 1), im(nfft / 2 + 1);
    auto mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto imel = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
    const double fLo = 60.0, fHi = std::min(7600.0, 0.475 * fsA);
    std::vector<double> edges(bands + 2);
    for (int i = 0; i < bands + 2; ++i) edges[i] = imel(mel(fLo) + (mel(fHi) - mel(fLo)) * i / (bands + 1));

    std::vector<std::vector<double>> frames;
    std::vector<double> energies;
    for (int s = 0; s + win <= static_cast<int>(y.size()); s += hop) {
        std::fill(buf.begin(), buf.end(), 0.0f);
        for (int i = 0; i < win; ++i)
            buf[i] = y[s + i] * static_cast<float>(0.54 - 0.46 * std::cos(2.0 * M_PI * i / (win - 1)));
        fft.forwardReal(buf.data(), re.data(), im.data());
        std::vector<double> logMel(bands, 0.0);
        for (int b = 0; b < bands; ++b) {
            double acc = 0.0;
            for (int k = 0; k <= nfft / 2; ++k) {
                const double f = k * fsA / nfft;
                double w = 0.0;
                if (f > edges[b] && f <= edges[b + 1]) w = (f - edges[b]) / (edges[b + 1] - edges[b]);
                else if (f > edges[b + 1] && f < edges[b + 2]) w = (edges[b + 2] - f) / (edges[b + 2] - edges[b + 1]);
                if (w > 0) acc += w * (re[k] * re[k] + im[k] * im[k]);
            }
            logMel[b] = std::log(acc + 1e-12);
        }
        std::vector<double> c(ceps, 0.0);
        for (int k = 0; k < ceps; ++k) {
            double acc = 0.0;
            for (int b = 0; b < bands; ++b) acc += logMel[b] * std::cos(M_PI * k * (b + 0.5) / bands);
            c[k] = acc;
        }
        frames.push_back(c);
        energies.push_back(frameEnergy(&y[s], win));
    }
    std::vector<float> mean(ceps - 1, 0.0f);
    if (frames.empty()) return mean;
    const double maxE = *std::max_element(energies.begin(), energies.end());
    int count = 0;
    std::vector<double> acc(ceps - 1, 0.0);
    for (size_t i = 0; i < frames.size(); ++i) {
        if (energies[i] < maxE * 1e-3 || energies[i] < 1e-10) continue;
        for (int k = 1; k < ceps; ++k) acc[k - 1] += frames[i][k];
        ++count;
    }
    if (count > 0)
        for (int k = 0; k < ceps - 1; ++k) mean[k] = static_cast<float>(acc[k] / count);
    return mean;
}

double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    return ab / std::sqrt(aa * bb + 1e-30);
}

int countClicks(const float* x, int n, double fs, std::vector<int>* positions, bool relaxed) {
    // A click is an *isolated* second-difference spike: much larger than the
    // local RMS over +-15 ms (which spans at least one pitch period, so regular
    // glottal pulses are not counted) with at most one comparable spike nearby.
    if (n < 8) return 0;
    std::vector<double> d2(n, 0.0), cum(n + 1, 0.0);
    for (int i = 1; i < n - 1; ++i) d2[i] = x[i + 1] - 2.0 * x[i] + x[i - 1];
    for (int i = 0; i < n; ++i) cum[i + 1] = cum[i] + d2[i] * d2[i];
    const int half = std::max(16, static_cast<int>(0.015 * fs));
    const int guard = std::max(2, static_cast<int>(0.001 * fs));
    int clicks = 0;
    int lastClick = -1000000;
    for (int i = 1; i < n - 1; ++i) {
        const double v = std::fabs(d2[i]);
        if (v < (relaxed ? 5e-4 : 2e-3)) continue;
        if (v < std::fabs(d2[i - 1]) || v < std::fabs(d2[i + 1])) continue;  // local maxima only
        const int lo = std::max(0, i - half), hi = std::min(n, i + half);
        const int glo = std::max(lo, i - guard), ghi = std::min(hi, i + guard + 1);
        const double e = (cum[hi] - cum[lo]) - (cum[ghi] - cum[glo]);
        const int cnt = (hi - lo) - (ghi - glo);
        if (cnt <= 0) continue;
        const double local = std::sqrt(e / cnt);
        if (v <= (relaxed ? 4.0 : 8.0) * local) continue;
        // Periodic pulses have >= 2 comparable neighbours within +-15 ms; a click
        // (or a step and its return) has at most one.
        int comparable = 0;
        for (int j = std::max(1, lo); j < std::min(n - 1, hi); ++j) {
            if (j >= glo && j < ghi) continue;
            const double a = std::fabs(d2[j]);
            if (a > v / 3.0 && a >= std::fabs(d2[j - 1]) && a >= std::fabs(d2[j + 1])) ++comparable;
        }
        if (!relaxed && comparable > 1) continue;
        if (i - lastClick > guard * 4) {
            ++clicks;
            if (positions) positions->push_back(i);
        }
        lastClick = i;
    }
    return clicks;
}

Comparison compare(const float* dryIn, const float* wetIn, int n, double fs) {
    const float* dry = dryIn;
    const float* wet = wetIn;
    Comparison c;
    // Pitch.
    const std::vector<float> fd = trackPitch(dry, n, fs);
    const std::vector<float> fw = trackPitch(wet, n, fs);
    c.f0DryHz = medianVoicedF0(fd);
    c.f0WetHz = medianVoicedF0(fw);
    std::vector<double> ratios, ld, lw;
    int dryVoiced = 0, both = 0;
    for (size_t i = 0; i < std::min(fd.size(), fw.size()); ++i) {
        if (fd[i] > 0) ++dryVoiced;
        if (fd[i] > 0 && fw[i] > 0) {
            ++both;
            ratios.push_back(12.0 * std::log2(fw[i] / fd[i]));
            ld.push_back(std::log(fd[i]));
            lw.push_back(std::log(fw[i]));
        }
    }
    if (!ratios.empty()) {
        std::vector<double> sorted = ratios;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        c.pitchShiftSemitones = sorted[sorted.size() / 2];
        // Contour similarity, excluding tracker gross errors (octave jumps etc.).
        std::vector<double> a, b;
        int gross = 0;
        for (size_t i = 0; i < ratios.size(); ++i) {
            if (std::fabs(ratios[i] - c.pitchShiftSemitones) > 3.0) {
                ++gross;
                continue;
            }
            a.push_back(ld[i]);
            b.push_back(lw[i]);
        }
        c.intonationCorrelation = pearson(a, b);
        c.f0GrossErrorRate = static_cast<double>(gross) / ratios.size();
    }
    c.voicingAgreement = dryVoiced > 0 ? static_cast<double>(both) / dryVoiced : 0.0;

    // Timbre proxies.
    c.formantRatio = estimateFormantRatio(dry, wet, n, fs);
    c.mfccCosine = cosineSimilarity(meanMfcc(dry, n, fs), meanMfcc(wet, n, fs));
    {
        const double maxHz = std::min(5000.0, 0.45 * std::min(fs, 16000.0));
        std::vector<float> ea = averageEnvelopeDb(dry, n, fs, maxHz, 200);
        std::vector<float> eb = averageEnvelopeDb(wet, n, fs, maxHz, 200);
        double ma = 0, mb = 0;
        for (int i = 0; i < 200; ++i) {
            ma += ea[i];
            mb += eb[i];
        }
        ma /= 200;
        mb /= 200;
        double acc = 0;
        for (int i = 0; i < 200; ++i) {
            const double d = (eb[i] - mb) - (ea[i] - ma);
            acc += d * d;
        }
        c.ltasDistanceDb = std::sqrt(acc / 200);
    }

    // Envelope / continuity, measured above 100 Hz (the engine removes rumble).
    std::vector<float> dryHp(dry, dry + n), wetHp(wet, wet + n);
    {
        Biquad h1 = Biquad::highpass(fs, 100.0, 0.5412), h2 = Biquad::highpass(fs, 100.0, 1.3066);
        Biquad g1 = h1, g2 = h2;
        for (int i = 0; i < n; ++i) {
            dryHp[i] = h2.process(h1.process(dryHp[i]));
            wetHp[i] = g2.process(g1.process(wetHp[i]));
        }
    }
    dry = dryHp.data();
    wet = wetHp.data();
    const int hop = static_cast<int>(0.010 * fs);
    std::vector<double> envD, envW, dbD, dbW;
    for (int s = 0; s + hop <= n; s += hop) {
        const double ed = frameEnergy(dry + s, hop), ew = frameEnergy(wet + s, hop);
        envD.push_back(std::sqrt(ed));
        envW.push_back(std::sqrt(ew));
        dbD.push_back(10.0 * std::log10(ed + 1e-20));
        dbW.push_back(10.0 * std::log10(ew + 1e-20));
    }
    // Dropouts are judged relative to each signal's own speech level (95th
    // percentile frame level), so level normalisation (AGC / limiter / gain)
    // is not mistaken for speech cutting out.
    auto p95 = [](std::vector<double> v) {
        if (v.empty()) return -120.0;
        std::sort(v.begin(), v.end());
        return v[static_cast<size_t>(0.95 * (v.size() - 1))];
    };
    const double refD = p95(dbD), refW = p95(dbW);
    for (size_t i = 0; i < dbD.size(); ++i) {
        const double relD = dbD[i] - refD;
        // Allow +-1 frame (10 ms) of onset/offset timing difference.
        double relW = dbW[i] - refW;
        if (i > 0) relW = std::max(relW, dbW[i - 1] - refW);
        if (i + 1 < dbW.size()) relW = std::max(relW, dbW[i + 1] - refW);
        if (dbD[i] > -50.0 && relD > -30.0 && relW < relD - 25.0) ++c.dropouts;
    }
    double bestCorr = -1.0;
    for (int lag = -3; lag <= 3; ++lag) {
        std::vector<double> a, b;
        for (size_t i = 0; i < envD.size(); ++i) {
            const long j = static_cast<long>(i) + lag;
            if (j < 0 || j >= static_cast<long>(envW.size())) continue;
            a.push_back(envD[i]);
            b.push_back(envW[j]);
        }
        bestCorr = std::max(bestCorr, pearson(a, b));
    }
    c.envelopeCorrelation = bestCorr;

    dry = dryIn;
    wet = wetIn;
    for (int i = 0; i < n; ++i) {
        c.wetPeak = std::max(c.wetPeak, static_cast<double>(std::fabs(wet[i])));
        if (std::fabs(wet[i]) >= 0.999f) ++c.clippedSamples;
    }
    c.dryRmsDb = rmsDb(dry, n);
    c.wetRmsDb = rmsDb(wet, n);

    std::vector<int> pw, pd;
    countClicks(wet, n, fs, &pw);
    countClicks(dry, n, fs, &pd, true);
    const int tol = static_cast<int>(0.012 * fs);
    for (int p : pw) {
        bool matched = false;
        for (int q : pd)
            if (std::abs(p - q) <= tol) {
                matched = true;
                break;
            }
        if (!matched) ++c.newClicks;
    }
    return c;
}

std::string toJson(const Comparison& c) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"f0DryHz\":%.1f,\"f0WetHz\":%.1f,\"pitchShiftSemitones\":%.2f,"
                  "\"intonationCorrelation\":%.3f,\"f0GrossErrorRate\":%.3f,\"voicingAgreement\":%.3f,\"formantRatio\":%.3f,"
                  "\"mfccCosine\":%.3f,\"ltasDistanceDb\":%.2f,\"envelopeCorrelation\":%.3f,"
                  "\"dropouts\":%d,\"clippedSamples\":%d,\"newClicks\":%d,\"wetPeak\":%.3f,"
                  "\"dryRmsDb\":%.1f,\"wetRmsDb\":%.1f}",
                  c.f0DryHz, c.f0WetHz, c.pitchShiftSemitones, c.intonationCorrelation, c.f0GrossErrorRate,
                  c.voicingAgreement,
                  c.formantRatio, c.mfccCosine, c.ltasDistanceDb, c.envelopeCorrelation, c.dropouts,
                  c.clippedSamples, c.newClicks, c.wetPeak, c.dryRmsDb, c.wetRmsDb);
    return buf;
}

}  // namespace analysis
}  // namespace voiceanon
