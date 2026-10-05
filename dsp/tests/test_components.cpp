// Unit tests for the individual DSP blocks.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "synth.h"
#include "test_framework.h"
#include "voiceanon/analysis.h"
#include "voiceanon/biquad.h"
#include "voiceanon/fft.h"
#include "voiceanon/limiter.h"
#include "voiceanon/noise_suppressor.h"
#include "voiceanon/pitch_tracker.h"
#include "voiceanon/presets.h"
#include "voiceanon/psola.h"

using namespace voiceanon;

namespace {

// Exactly periodic vowel-like signal: harmonics shaped by a formant envelope.
std::vector<float> harmonicVowel(int n, double fs, double f0, double formantScale = 1.0) {
    const double formants[3] = {730 * formantScale, 1090 * formantScale, 2440 * formantScale};
    const double bws[3] = {90, 110, 160};
    std::vector<float> x(n, 0.0f);
    for (int h = 1; h * f0 < std::min(5000.0, 0.45 * fs); ++h) {
        const double f = h * f0;
        double a = 0.0;
        for (int k = 0; k < 3; ++k) a += 1.0 / (1.0 + std::pow((f - formants[k]) / bws[k], 2.0)) / (k + 1);
        a += 0.02;
        const double ph = 0.37 * h * h;
        for (int i = 0; i < n; ++i) x[i] += static_cast<float>(a * std::sin(2.0 * M_PI * f * i / fs + ph));
    }
    double e = 0.0;
    for (float v : x) e += static_cast<double>(v) * v;
    const double g = 0.1 / std::sqrt(e / n);
    for (float& v : x) v = static_cast<float>(v * g);
    return x;
}

std::vector<float> runPsola(const std::vector<float>& x, double fs, float pitch, float formant, int* latency) {
    PitchTracker tr(fs);
    PsolaShifter ps(fs);
    *latency = ps.latency();
    std::vector<float> out(x.size(), 0.0f);
    const int block = 128;
    for (size_t i = 0; i + block <= x.size(); i += block) {
        tr.push(&x[i], block);
        ps.process(&x[i], &out[i], block, tr, pitch, formant, 1.0f, 0.0f);
    }
    return out;
}

}  // namespace

TEST(fft_roundtrip) {
    for (int n : {16, 256, 512, 1024}) {
        Fft fft(n);
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> u(-1, 1);
        std::vector<float> x(n), y(n), re(n / 2 + 1), im(n / 2 + 1);
        for (float& v : x) v = u(rng);
        fft.forwardReal(x.data(), re.data(), im.data());
        fft.inverseReal(re.data(), im.data(), y.data());
        double err = 0;
        for (int i = 0; i < n; ++i) err = std::max(err, static_cast<double>(std::fabs(x[i] - y[i])));
        CHECK_LE(err, 1e-5);
    }
    // A pure tone lands in the right bin.
    Fft fft(512);
    std::vector<float> x(512), re(257), im(257);
    for (int i = 0; i < 512; ++i) x[i] = static_cast<float>(std::cos(2 * M_PI * 32 * i / 512.0));
    fft.forwardReal(x.data(), re.data(), im.data());
    CHECK_NEAR(re[32], 256.0, 1e-2);
}

TEST(biquad_highpass_removes_dc_and_rumble) {
    const double fs = 48000;
    Biquad hp = Biquad::highpass(fs, 70, 0.7071);
    float y = 0;
    for (int i = 0; i < 48000; ++i) y = hp.process(0.5f);
    CHECK_LE(std::fabs(y), 1e-4);
    std::vector<float> s = testing::sine(48000, fs, 1000, 0.5);
    Biquad hp2 = Biquad::highpass(fs, 70, 0.7071);
    for (float& v : s) v = hp2.process(v);
    CHECK_NEAR(analysis::rmsDb(s.data() + 24000, 24000), 20 * std::log10(0.5 / std::sqrt(2.0)), 0.1);
}

TEST(noise_suppressor_is_transparent_when_off) {
    for (double fs : {16000.0, 48000.0}) {
        NoiseSuppressor ns(fs);
        ns.setAmount(0.0f);
        const int hop = ns.hop(), lat = ns.latency();
        std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 3);
        x.resize((x.size() / hop) * hop);
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); i += hop) ns.processHop(&x[i], &y[i]);
        double err = 0;
        for (size_t i = lat; i < x.size(); ++i) err = std::max(err, static_cast<double>(std::fabs(y[i] - x[i - lat])));
        CHECK_LE(err, 1e-5);
    }
}

TEST(noise_suppressor_reduces_stationary_noise) {
    const double fs = 48000;
    NoiseSuppressor ns(fs);
    ns.setAmount(1.0f);
    const int hop = ns.hop();
    std::vector<float> x = testing::pinkNoise(static_cast<int>(fs * 3), -40, 5);
    x.resize((x.size() / hop) * hop);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); i += hop) ns.processHop(&x[i], &y[i]);
    const int from = static_cast<int>(fs * 1.5);
    const double reduction = analysis::rmsDb(x.data() + from, static_cast<int>(x.size()) - from) -
                             analysis::rmsDb(y.data() + from, static_cast<int>(y.size()) - from);
    vt::note("pink noise attenuation at amount=1: %.1f dB", reduction);
    CHECK_GE(reduction, 12.0);
    CHECK(!ns.voiceActive());
}

TEST(noise_suppressor_vad_detects_speech) {
    const double fs = 16000;
    NoiseSuppressor ns(fs);
    ns.setAmount(0.5f);
    const int hop = ns.hop();
    std::vector<float> noise = testing::pinkNoise(static_cast<int>(fs * 1.0), -50, 9);
    std::vector<float> speech = testing::synthesizeSpeech("aa aa aa", testing::maleVoice(), fs, 4, -20);
    std::vector<float> x = noise;
    x.insert(x.end(), speech.begin(), speech.end());
    x.resize((x.size() / hop) * hop);
    std::vector<float> y(x.size());
    int noiseVad = 0, noiseFrames = 0, speechVad = 0, speechFrames = 0;
    for (size_t i = 0; i < x.size(); i += hop) {
        ns.processHop(&x[i], &y[i]);
        if (i > fs * 0.4 && i < noise.size()) {
            ++noiseFrames;
            noiseVad += ns.voiceActive();
        }
        if (i > noise.size() + fs * 0.1 && i < noise.size() + speech.size() - fs * 0.1) {
            ++speechFrames;
            speechVad += ns.voiceActive();
        }
    }
    vt::note("VAD: noise frames flagged %d/%d, speech frames flagged %d/%d", noiseVad, noiseFrames, speechVad,
             speechFrames);
    CHECK_LE(noiseVad, noiseFrames * 0.05);
    CHECK_GE(speechVad, speechFrames * 0.95);
}

TEST(pitch_tracker_sines_and_voices) {
    for (double fs : {16000.0, 44100.0, 48000.0}) {
        for (double f : {80.0, 120.0, 200.0, 350.0}) {
            PitchTracker tr(fs);
            std::vector<float> x = testing::sine(static_cast<int>(fs * 0.5), fs, f, 0.3);
            tr.push(x.data(), static_cast<int>(x.size()));
            CHECK(tr.voiced());
            CHECK_NEAR(tr.f0(), f, f * 0.01);
        }
    }
    for (const auto& v : {testing::maleVoice(), testing::femaleVoice()}) {
        const double fs = 48000;
        std::vector<float> x = testing::synthesizeSpeech("aa aa aa aa", v, fs, 2);
        PitchTracker tr(fs);
        std::vector<float> est;
        for (size_t i = 0; i + 256 <= x.size(); i += 256) {
            tr.push(&x[i], 256);
            if (tr.voiced()) est.push_back(tr.f0());
        }
        std::sort(est.begin(), est.end());
        CHECK(!est.empty());
        if (!est.empty()) CHECK_NEAR(est[est.size() / 2], v.f0Hz, v.f0Hz * 0.12);
    }
    // Silence is unvoiced.
    PitchTracker tr(48000);
    std::vector<float> z(24000, 0.0f);
    tr.push(z.data(), static_cast<int>(z.size()));
    CHECK(!tr.voiced());
}

TEST(psola_identity_is_transparent) {
    // pitch = formant = 1 on an exactly periodic signal must reproduce the input.
    const double fs = 48000;
    const std::vector<float> x = harmonicVowel(static_cast<int>(fs * 1.0), fs, 120.0);  // period 400 samples
    int lat = 0;
    const std::vector<float> y = runPsola(x, fs, 1.0f, 1.0f, &lat);
    double err = 0, sig = 0;
    for (size_t i = static_cast<size_t>(fs * 0.3); i + lat < y.size(); ++i) {
        const double d = y[i + lat] - x[i];
        err += d * d;
        sig += static_cast<double>(x[i]) * x[i];
    }
    const double snr = 10 * std::log10(sig / (err + 1e-30));
    vt::note("PSOLA identity SNR: %.1f dB (latency %d samples = %.2f ms)", snr, lat, 1000.0 * lat / fs);
    CHECK_GE(snr, 30.0);
}

TEST(psola_pitch_accuracy) {
    const double fs = 48000;
    for (double f0 : {110.0, 210.0}) {
        const std::vector<float> x = harmonicVowel(static_cast<int>(fs * 1.5), fs, f0);
        for (double semis : {-3.0, -1.5, 1.5, 3.0, 4.5}) {
            int lat = 0;
            const float ratio = static_cast<float>(std::pow(2.0, semis / 12.0));
            const std::vector<float> y = runPsola(x, fs, ratio, 1.0f, &lat);
            const int n = static_cast<int>(x.size()) - lat;
            const analysis::Comparison c = analysis::compare(x.data(), y.data() + lat, n, fs);
            vt::note("F0 %.0f Hz, target %+.1f st -> measured %+.2f st", f0, semis, c.pitchShiftSemitones);
            CHECK_NEAR(c.pitchShiftSemitones, semis, 0.25);
        }
    }
}

TEST(psola_formant_accuracy) {
    const double fs = 48000;
    const std::vector<float> x = harmonicVowel(static_cast<int>(fs * 1.5), fs, 115.0);
    for (double r : {0.88, 0.94, 1.06, 1.12, 1.18}) {
        int lat = 0;
        const std::vector<float> y = runPsola(x, fs, 1.0f, static_cast<float>(r), &lat);
        const int n = static_cast<int>(x.size()) - lat;
        const analysis::Comparison c = analysis::compare(x.data(), y.data() + lat, n, fs);
        vt::note("formant target x%.2f -> measured x%.3f, F0 shift %+.2f st", r, c.formantRatio,
                 c.pitchShiftSemitones);
        CHECK_NEAR(c.formantRatio, r, 0.03);
        CHECK_NEAR(c.pitchShiftSemitones, 0.0, 0.2);  // formant shift must not move F0
    }
}

TEST(limiter_never_exceeds_ceiling) {
    const double fs = 48000;
    Limiter lim(fs);
    std::vector<float> x = testing::whiteNoise(static_cast<int>(fs), 0.0, 3);  // ~0 dBFS RMS, peaks >> 1
    for (float& v : x) v *= 3.0f;
    std::vector<float> s = testing::sine(static_cast<int>(fs), fs, 200, 2.5);
    x.insert(x.end(), s.begin(), s.end());
    lim.process(x.data(), static_cast<int>(x.size()));
    float peak = 0;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    CHECK_LE(peak, lim.ceiling() + 1e-6);

    // Quiet material passes bit-exactly (only delayed).
    Limiter l2(fs);
    std::vector<float> q = testing::sine(4800, fs, 440, 0.25);
    std::vector<float> y = q;
    l2.process(y.data(), static_cast<int>(y.size()));
    double err = 0;
    for (size_t i = l2.latency(); i < y.size(); ++i) err = std::max(err, static_cast<double>(std::fabs(y[i] - q[i - l2.latency()])));
    CHECK_LE(err, 1e-7);
}

TEST(presets_are_bounded_and_ordered) {
    const Params n = presetParams(Preset::Natural), b = presetParams(Preset::Balanced),
                 s = presetParams(Preset::Strong);
    CHECK(n.pitchSemitones < b.pitchSemitones && b.pitchSemitones < s.pitchSemitones);
    CHECK(n.formantPercent < b.formantPercent && b.formantPercent < s.formantPercent);
    for (float st = 0.0f; st <= 1.0f; st += 0.05f) {
        const Params p = paramsForStrength(st);
        CHECK_LE(p.pitchSemitones, 4.5);
        CHECK_LE(p.formantPercent, 16.0);
        CHECK_GE(p.intonation, 0.75);
    }
    Preset pr;
    CHECK(parsePreset("natural", &pr) && pr == Preset::Natural);
    CHECK(parsePreset("strong", &pr) && pr == Preset::Strong);
    CHECK(!parsePreset("robot", &pr));
}

TEST(analysis_click_detector) {
    const double fs = 48000;
    std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 11);
    const int base = analysis::countClicks(x.data(), static_cast<int>(x.size()), fs);
    // Insert three hard discontinuities.
    for (int k = 1; k <= 3; ++k) {
        const size_t p = x.size() * k / 4;
        for (size_t i = p; i < p + 200; ++i) x[i] += 0.3f;
    }
    const int withClicks = analysis::countClicks(x.data(), static_cast<int>(x.size()), fs);
    vt::note("clicks: clean %d, with 3 injected steps %d", base, withClicks);
    CHECK_LE(base, 1);
    CHECK_GE(withClicks - base, 3);
}
