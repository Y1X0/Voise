// End-to-end tests of the streaming engine on reproducible synthetic material.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "synth.h"
#include "test_framework.h"
#include "voiceanon/analysis.h"
#include "voiceanon/engine.h"
#include "voiceanon/presets.h"

using namespace voiceanon;

namespace {

struct Run {
    std::vector<float> wet;  // latency-aligned with the input
    std::vector<float> raw;  // unaligned engine output
    int latency = 0;
    Metrics metrics;
};

// Streams `x` through a fresh engine using the given block-size pattern.
Run runEngine(const std::vector<float>& x, double fs, const Params& p, const std::vector<int>& blocks = {192}) {
    Engine e(fs);
    e.setParams(p);
    e.reset();
    Run r;
    r.latency = e.latencySamples();
    std::vector<float> in = x;
    in.resize(x.size() + r.latency + 4096, 0.0f);
    r.raw.assign(in.size(), 0.0f);
    size_t i = 0, b = 0;
    while (i < in.size()) {
        const int n = static_cast<int>(std::min<size_t>(blocks[b++ % blocks.size()], in.size() - i));
        e.process(&in[i], &r.raw[i], n);
        i += n;
    }
    r.wet.assign(r.raw.begin() + r.latency, r.raw.begin() + r.latency + x.size());
    r.metrics = e.metrics();
    return r;
}

bool allFinite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

float peak(const std::vector<float>& v) {
    float p = 0;
    for (float x : v) p = std::max(p, std::fabs(x));
    return p;
}

Params fixedUp(Preset preset) {
    Params p = presetParams(preset);
    p.direction = Direction::Up;
    return p;
}

double expectedSemitones(const Params& p, int sign) { return sign * p.pitchSemitones; }
double expectedFormant(const Params& p, int sign) {
    const double r = 1.0 + p.formantPercent / 100.0;
    return sign > 0 ? r : 1.0 / r;
}

void logComparison(const char* label, const analysis::Comparison& c) {
    vt::note("%-34s pitch %+5.2f st | formant x%.3f | env.corr %.3f | inton.corr %.3f | mfccCos %.3f | "
             "LTAS %.2f dB | dropouts %d | newClicks %d | peak %.3f",
             label, c.pitchShiftSemitones, c.formantRatio, c.envelopeCorrelation, c.intonationCorrelation,
             c.mfccCosine, c.ltasDistanceDb, c.dropouts, c.newClicks, c.wetPeak);
}

}  // namespace

TEST(engine_silence_in_silence_out) {
    for (double fs : {16000.0, 48000.0}) {
        std::vector<float> z(static_cast<size_t>(fs * 3), 0.0f);
        const Run r = runEngine(z, fs, presetParams(Preset::Strong));
        CHECK(allFinite(r.raw));
        CHECK_LE(peak(r.raw), 1e-9);
    }
}

TEST(engine_speech_male_female_arabic_english) {
    struct Case {
        const char* label;
        testing::VoiceProfile voice;
        std::string text;
        Direction dir;
    };
    const Case cases[] = {
        {"male / Arabic-like / up", testing::maleVoice(), testing::arabicText(), Direction::Up},
        {"male / English-like / up", testing::maleVoice(), testing::englishText(), Direction::Up},
        {"female / Arabic-like / down", testing::femaleVoice(), testing::arabicText(), Direction::Down},
        {"female / English-like / down", testing::femaleVoice(), testing::englishText(), Direction::Down},
        {"female / English-like / up", testing::femaleVoice(), testing::englishText(), Direction::Up},
    };
    const double fs = 48000;
    uint32_t seed = 100;
    for (const Case& c : cases) {
        for (Preset preset : {Preset::Natural, Preset::Balanced, Preset::Strong}) {
            const std::vector<float> x =
                testing::pad(testing::synthesizeSpeech(c.text, c.voice, fs, seed++, -20.0), fs, 0.3, 0.3);
            Params p = presetParams(preset);
            p.direction = c.dir;
            p.intonation = 1.0f;  // isolate pitch/formant accuracy
            const Run r = runEngine(x, fs, p);
            const analysis::Comparison cmp = analysis::compare(x.data(), r.wet.data(), static_cast<int>(x.size()), fs);
            const int sign = c.dir == Direction::Down ? -1 : 1;
            char label[96];
            std::snprintf(label, sizeof(label), "%s [%s]", c.label, presetName(preset));
            logComparison(label, cmp);
            CHECK(allFinite(r.raw));
            CHECK_NEAR(cmp.pitchShiftSemitones, expectedSemitones(p, sign), 0.5);
            CHECK_NEAR(cmp.formantRatio, expectedFormant(p, sign), 0.04);
            CHECK_GE(cmp.envelopeCorrelation, 0.85);  // syllabic rhythm / content timing preserved
            CHECK_GE(cmp.intonationCorrelation, 0.85);  // melody preserved (not monotone / robotic)
            CHECK(cmp.dropouts == 0);                   // speech never cut out
            CHECK(cmp.clippedSamples == 0);
            CHECK_LE(cmp.newClicks, 1);
        }
    }
}

TEST(engine_loud_speech_never_clips) {
    const double fs = 48000;
    std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 7, -6.0);
    vt::note("input peak %.2f (%.1f dBFS)", peak(x), 20 * std::log10(peak(x)));
    for (float gain : {1.0f, 3.0f}) {
        std::vector<float> in = x;
        for (float& v : in) v *= gain;
        const Run r = runEngine(in, fs, fixedUp(Preset::Strong));
        CHECK(allFinite(r.raw));
        CHECK_LE(peak(r.raw), 0.8913 + 1e-4);
        const analysis::Comparison c = analysis::compare(in.data(), r.wet.data(), static_cast<int>(in.size()), fs);
        vt::note("gain x%.0f: output peak %.3f, limiter %.1f dB, env.corr %.3f", gain, peak(r.raw),
                 r.metrics.limiterGainDb, c.envelopeCorrelation);
        CHECK_GE(c.envelopeCorrelation, 0.8);
        CHECK(c.dropouts == 0);
    }
}

TEST(engine_quiet_speech_is_normalised) {
    const double fs = 48000;
    const std::vector<float> x =
        testing::pad(testing::synthesizeSpeech(testing::arabicText(), testing::femaleVoice(), fs, 8, -45.0), fs, 0.5, 0.2);
    const Run r = runEngine(x, fs, fixedUp(Preset::Balanced));
    const analysis::Comparison c = analysis::compare(x.data(), r.wet.data(), static_cast<int>(x.size()), fs);
    vt::note("quiet speech: in %.1f dBFS -> out %.1f dBFS (AGC %.1f dB)", c.dryRmsDb, c.wetRmsDb, r.metrics.agcGainDb);
    CHECK_GE(c.wetRmsDb - c.dryRmsDb, 6.0);
    CHECK(c.dropouts == 0);
    CHECK_NEAR(c.pitchShiftSemitones, presetParams(Preset::Balanced).pitchSemitones, 0.6);
}

TEST(engine_background_noise) {
    const double fs = 48000;
    for (const auto& noiseKind : {"pink", "white"}) {
        std::vector<float> speech = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 9, -20.0);
        std::vector<float> x = testing::pad(speech, fs, 1.5, 0.3);
        const std::vector<float> noise = std::string(noiseKind) == "pink"
                                             ? testing::pinkNoise(static_cast<int>(x.size()), -20, 21)
                                             : testing::whiteNoise(static_cast<int>(x.size()), -20, 22);
        // 10 dB SNR relative to the speech part.
        std::vector<float> noisy = speech;
        testing::mixAtSnr(noisy, noise, 10.0);
        const double noiseGain = std::sqrt([&] {
            double es = 0, en = 0;
            for (size_t i = 0; i < speech.size(); ++i) {
                es += static_cast<double>(speech[i]) * speech[i];
                en += static_cast<double>(noise[i]) * noise[i];
            }
            return es / en;
        }()) * std::pow(10.0, -10.0 / 20.0);
        for (size_t i = 0; i < x.size(); ++i) x[i] += static_cast<float>(noiseGain * noise[i]);

        Params p = fixedUp(Preset::Balanced);
        p.noiseSuppression = 0.7f;
        const Run r = runEngine(x, fs, p);
        const int noiseOnly = static_cast<int>(fs * 1.0);
        const int from = static_cast<int>(fs * 0.6);
        const double inDb = analysis::rmsDb(x.data() + from, noiseOnly - from);
        const double outDb = analysis::rmsDb(r.wet.data() + from, noiseOnly - from);
        const analysis::Comparison c = analysis::compare(x.data(), r.wet.data(), static_cast<int>(x.size()), fs);
        vt::note("%s noise @10 dB SNR: noise-only %.1f -> %.1f dBFS (%.1f dB less), pitch %+.2f st, env.corr %.3f",
                 noiseKind, inDb, outDb, inDb - outDb, c.pitchShiftSemitones, c.envelopeCorrelation);
        CHECK(allFinite(r.raw));
        CHECK_GE(inDb - outDb, 8.0);
        CHECK_NEAR(c.pitchShiftSemitones, p.pitchSemitones, 0.7);
        CHECK_GE(c.envelopeCorrelation, 0.75);
    }
}

TEST(engine_output_independent_of_block_size) {
    const double fs = 48000;
    const std::vector<float> x = testing::synthesizeSpeech(testing::arabicText(), testing::maleVoice(), fs, 12, -20.0);
    const Params p = presetParams(Preset::Balanced);
    const Run ref = runEngine(x, fs, p, {192});
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> u(1, 2000);
    std::vector<int> randomBlocks(500);
    for (int& b : randomBlocks) b = u(rng);
    for (const std::vector<int>& blocks :
         std::vector<std::vector<int>>{{1}, {7}, {64}, {240}, {480}, {1024}, {4096}, randomBlocks}) {
        const Run r = runEngine(x, fs, p, blocks);
        double maxDiff = 0;
        for (size_t i = 0; i < ref.raw.size(); ++i) maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(ref.raw[i] - r.raw[i])));
        CHECK(maxDiff == 0.0);
    }
}

TEST(engine_survives_lost_frames) {
    const double fs = 48000;
    const std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::femaleVoice(), fs, 13, -20.0);
    Engine e(fs);
    e.setParams(fixedUp(Preset::Balanced));
    e.reset();
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<float> out(x.size(), 0.0f);
    const int block = 192;
    int lostBlocks = 0;
    for (size_t i = 0; i + block <= x.size(); i += block) {
        const bool lose = u(rng) < 0.10f;  // 10 % of callbacks lose their input
        lostBlocks += lose;
        e.process(lose ? nullptr : &x[i], &out[i], block);
    }
    const Metrics m = e.metrics();
    vt::note("lost %d blocks (%llu frames reported), output peak %.3f", lostBlocks,
             static_cast<unsigned long long>(m.lostFrames), peak(out));
    CHECK(allFinite(out));
    CHECK_LE(peak(out), 0.8913 + 1e-4);
    CHECK(m.lostFrames == static_cast<uint64_t>(lostBlocks) * block);
    CHECK_GE(analysis::rmsDb(out.data() + out.size() / 2, static_cast<int>(out.size() / 2)), -40.0);  // keeps running
}

TEST(engine_parameter_changes_are_click_free) {
    const double fs = 48000;
    const std::vector<float> x =
        testing::synthesizeSpeech(testing::arabicText() + " " + testing::englishText(), testing::maleVoice(), fs, 14, -20.0);
    Engine e(fs);
    e.reset();
    std::vector<float> out(x.size() + e.latencySamples(), 0.0f);
    std::vector<float> in = x;
    in.resize(out.size(), 0.0f);
    const int block = 240;
    int k = 0;
    for (size_t i = 0; i + block <= in.size(); i += block, ++k) {
        if (k % 30 == 0) {  // every 150 ms: switch preset / direction, toggle bypass every 5th switch
            Params p = presetParams(static_cast<Preset>((k / 30) % 3));
            p.direction = ((k / 30) % 2) ? Direction::Up : Direction::Down;
            p.anonymize = ((k / 30) % 5) != 4;
            p.noiseSuppression = 0.2f * ((k / 30) % 5);
            p.outputGainDb = static_cast<float>(((k / 30) % 3) * 3);
            e.setParams(p);
        }
        e.process(&in[i], &out[i], block);
    }
    const std::vector<float> wet(out.begin() + e.latencySamples(), out.begin() + e.latencySamples() + x.size());
    const analysis::Comparison c = analysis::compare(x.data(), wet.data(), static_cast<int>(x.size()), fs);
    vt::note("rapid preset/direction/bypass/gain changes: newClicks %d, dropouts %d, peak %.3f", c.newClicks,
             c.dropouts, c.wetPeak);
    CHECK(allFinite(out));
    CHECK_LE(c.newClicks, 2);
    CHECK(c.dropouts == 0);
}

TEST(engine_bypass_is_exact_and_latency_matches) {
    for (double fs : {16000.0, 44100.0, 48000.0}) {
        const std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 15, -26.0);
        Params p = presetParams(Preset::Balanced);
        p.anonymize = false;
        const Run r = runEngine(x, fs, p, {97});
        double err = 0;
        for (size_t i = 0; i < x.size(); ++i) err = std::max(err, static_cast<double>(std::fabs(r.wet[i] - x[i])));
        vt::note("fs %.0f: latency %d samples = %.2f ms, bypass max error %.2e", fs, r.latency, 1000.0 * r.latency / fs, err);
        CHECK(err == 0.0);
        CHECK_LE(1000.0 * r.latency / fs, 40.0);
    }
}

TEST(engine_wet_is_time_aligned_with_dry) {
    const double fs = 48000;
    const std::vector<float> x =
        testing::pad(testing::synthesizeSpeech(testing::arabicText(), testing::maleVoice(), fs, 16, -20.0), fs, 0.2, 0.2);
    const Run r = runEngine(x, fs, fixedUp(Preset::Strong));
    const int hop = static_cast<int>(fs * 0.005);
    std::vector<double> a, b;
    for (size_t i = 0; i + hop <= x.size(); i += hop) {
        a.push_back(std::sqrt(std::pow(10.0, analysis::rmsDb(&x[i], hop) / 10.0)));
        b.push_back(std::sqrt(std::pow(10.0, analysis::rmsDb(&r.wet[i], hop) / 10.0)));
    }
    int bestLag = 0;
    double best = -1;
    for (int lag = -8; lag <= 8; ++lag) {
        double s = 0;
        for (size_t i = 8; i + 8 < a.size(); ++i) s += a[i] * b[i + lag];
        if (s > best) {
            best = s;
            bestLag = lag;
        }
    }
    vt::note("residual wet/dry envelope lag: %d ms", bestLag * 5);
    CHECK_LE(std::abs(bestLag * 5), 10);
}

TEST(engine_sanitizes_nan_inf_and_huge_input) {
    const double fs = 48000;
    std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 17, -20.0);
    for (size_t i = 1000; i < x.size(); i += 7919) x[i] = std::nanf("");
    for (size_t i = 2000; i < x.size(); i += 10007) x[i] = INFINITY;
    for (size_t i = 3000; i < x.size(); i += 12011) x[i] = 1e9f;
    const Run r = runEngine(x, fs, presetParams(Preset::Balanced));
    CHECK(allFinite(r.raw));
    CHECK_LE(peak(r.raw), 0.8913 + 1e-4);
    CHECK(r.metrics.sanitizedSamples > 0);
}

TEST(engine_process_never_allocates) {
    const double fs = 48000;
    Engine e(fs);
    e.setParams(presetParams(Preset::Strong));
    e.reset();
    const std::vector<float> x = testing::synthesizeSpeech(testing::arabicText(), testing::femaleVoice(), fs, 18, -20.0);
    std::vector<float> out(x.size());
    const long before = vt::g_allocations.load();
    for (size_t i = 0; i + 192 <= x.size(); i += 192) {
        e.process(&x[i], &out[i], 192);
        if (i % 9600 == 0) e.setParams(presetParams(Preset::Natural));  // parameter updates are lock/alloc free too
        (void)e.metrics();
    }
    e.process(nullptr, out.data(), 192);  // loss concealment path
    const long allocs = vt::g_allocations.load() - before;
    vt::note("heap allocations during %zu samples of processing: %ld", x.size(), allocs);
    CHECK(allocs == 0);
}

TEST(engine_sample_rates) {
    for (double fs : {16000.0, 22050.0, 32000.0, 44100.0, 48000.0}) {
        const std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 19, -20.0);
        Params p = fixedUp(Preset::Balanced);
        p.intonation = 1.0f;
        const Run r = runEngine(x, fs, p);
        const analysis::Comparison c = analysis::compare(x.data(), r.wet.data(), static_cast<int>(x.size()), fs);
        vt::note("fs %5.0f Hz: latency %.1f ms, pitch %+.2f st, formant x%.3f", fs, 1000.0 * r.latency / fs,
                 c.pitchShiftSemitones, c.formantRatio);
        CHECK(allFinite(r.raw));
        CHECK_NEAR(c.pitchShiftSemitones, p.pitchSemitones, 0.5);
        CHECK_NEAR(c.formantRatio, 1.0 + p.formantPercent / 100.0, 0.05);
    }
}

TEST(engine_auto_direction_avoids_child_and_monster_regions) {
    const double fs = 48000;
    for (const auto& v : {testing::maleVoice(), testing::femaleVoice()}) {
        const std::vector<float> x = testing::synthesizeSpeech(
            testing::englishText() + " " + testing::arabicText(), v, fs, 20, -20.0);
        Params p = presetParams(Preset::Balanced);
        p.direction = Direction::Auto;
        const Run r = runEngine(x, fs, p);
        const int expectedDir = v.f0Hz < 165 ? 1 : -1;
        // Measure on the second half (after the direction has latched).
        const size_t h = x.size() / 2;
        const analysis::Comparison c =
            analysis::compare(x.data() + h, r.wet.data() + h, static_cast<int>(x.size() - h), fs);
        vt::note("%s (F0 %.0f Hz): latched direction %+d, mean F0 %.0f Hz, measured %+.2f st, formant x%.3f",
                 v.name.c_str(), v.f0Hz, r.metrics.direction, r.metrics.meanF0, c.pitchShiftSemitones, c.formantRatio);
        CHECK(r.metrics.direction == expectedDir);
        CHECK(c.pitchShiftSemitones * expectedDir > 1.0);
        // Output F0 stays inside the adult range.
        CHECK(c.f0WetHz > 85.0 && c.f0WetHz < 260.0);
    }
}

TEST(engine_capture_is_aligned) {
    const double fs = 16000;
    Engine e(fs);
    Params p = presetParams(Preset::Balanced);
    p.anonymize = false;  // wet == limited dry when bypassed? no: capture stores the wet path, so check dry only
    e.setParams(p);
    e.reset();
    CHECK(e.startCapture(1.0));
    const std::vector<float> x = testing::synthesizeSpeech(testing::englishText(), testing::maleVoice(), fs, 21, -20.0);
    std::vector<float> out(x.size());
    for (size_t i = 0; i + 160 <= x.size() && !e.captureReady(); i += 160) e.process(&x[i], &out[i], 160);
    CHECK(e.captureReady());
    std::vector<float> dry, wet;
    const int n = e.readCapture(dry, wet);
    CHECK(n == static_cast<int>(fs));
    // Captured dry = input delayed by the in-quantum latency (FIFO pre-fill excluded).
    const int inner = e.latencySamples() - (e.quantum() - 1);
    double err = 0;
    for (int i = inner; i < n; ++i) err = std::max(err, static_cast<double>(std::fabs(dry[i] - x[i - inner])));
    CHECK_LE(err, 1e-7);
    CHECK(!e.captureReady());
}

TEST(engine_long_run_stability_and_speed) {
    const double fs = 48000;
    const std::vector<float> speech =
        testing::synthesizeSpeech(testing::arabicText() + " _ " + testing::englishText(), testing::maleVoice(), fs, 22, -20.0);
    std::vector<float> noise = testing::pinkNoise(static_cast<int>(fs * 7), -50, 23);
    Engine e(fs);
    e.setParams(fixedUp(Preset::Balanced));
    e.reset();
    const double seconds = 120.0;
    const size_t total = static_cast<size_t>(seconds * fs);
    std::vector<float> in(192), out(192);
    double firstDb = 0, lastDb = 0, sum = 0;
    size_t count = 0;
    bool finite = true;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t pos = 0; pos < total; pos += 192) {
        for (int i = 0; i < 192; ++i) {
            const size_t k = pos + i;
            in[i] = speech[k % speech.size()] + noise[k % noise.size()];
        }
        e.process(in.data(), out.data(), 192);
        for (float v : out) {
            finite = finite && std::isfinite(v);
            sum += static_cast<double>(v) * v;
        }
        count += 192;
        if (count >= static_cast<size_t>(fs * 10)) {
            const double db = 10 * std::log10(sum / count + 1e-20);
            if (pos < fs * 11) firstDb = db;
            lastDb = db;
            sum = 0;
            count = 0;
        }
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    vt::note("%.0f s of audio processed in %.2f s (real-time factor %.4f, %.1f%% of one host core)", seconds, elapsed,
             elapsed / seconds, 100.0 * elapsed / seconds);
    vt::note("output level first 10 s %.1f dBFS, last 10 s %.1f dBFS", firstDb, lastDb);
    CHECK(finite);
    CHECK_NEAR(lastDb, firstDb, 3.0);
    CHECK_LE(elapsed / seconds, 0.25);
}

TEST(engine_experimental_dimensions_are_safe) {
    const double fs = 48000;
    const std::vector<float> x =
        testing::synthesizeSpeech(testing::arabicText() + " " + testing::englishText(), testing::maleVoice(), fs, 30, -20.0);
    const Params base = fixedUp(Preset::Balanced);
    const Run ref = runEngine(x, fs, base);
    struct Dim {
        const char* name;
        void (*apply)(Params&);
    };
    const Dim dims[] = {
        {"spectral reshape 6 dB", [](Params& p) { p.spectralReshapeDb = 6.0f; }},
        {"pitch drift 1 st", [](Params& p) { p.pitchDriftSemitones = 1.0f; }},
        {"formant jitter 4 %", [](Params& p) { p.formantJitterPercent = 4.0f; }},
        {"dynamics flatten 1.0", [](Params& p) { p.dynamicsFlatten = 1.0f; }},
        {"all combined", [](Params& p) {
             p.spectralReshapeDb = 6.0f;
             p.pitchDriftSemitones = 1.0f;
             p.formantJitterPercent = 4.0f;
             p.dynamicsFlatten = 1.0f;
         }},
    };
    for (const Dim& d : dims) {
        Params p = base;
        d.apply(p);
        p.variationSeed = 7;
        const Run a = runEngine(x, fs, p, {192});
        const Run b = runEngine(x, fs, p, {1, 1000, 37});
        double maxDiff = 0;
        for (size_t i = 0; i < a.raw.size(); ++i) maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(a.raw[i] - b.raw[i])));
        const analysis::Comparison c = analysis::compare(x.data(), a.wet.data(), static_cast<int>(x.size()), fs);
        vt::note("%-22s latency %d (base %d), env.corr %.3f, dropouts %d, newClicks %d, peak %.3f", d.name, a.latency,
                 ref.latency, c.envelopeCorrelation, c.dropouts, c.newClicks, c.wetPeak);
        CHECK(allFinite(a.raw));
        CHECK(a.latency == ref.latency);  // no added latency
        CHECK(maxDiff == 0.0);            // still deterministic and block-size independent
        CHECK_LE(peak(a.raw), 0.8913 + 1e-4);
        CHECK(c.dropouts == 0);
        CHECK_LE(c.newClicks, 2);
    }
    // Zero heap allocations with every experimental dimension active (and changing).
    Engine e(fs);
    Params p = base;
    p.spectralReshapeDb = 6.0f;
    p.pitchDriftSemitones = 1.0f;
    p.formantJitterPercent = 4.0f;
    p.dynamicsFlatten = 1.0f;
    e.setParams(p);
    e.reset();
    std::vector<float> out(x.size());
    const long before = vt::g_allocations.load();
    for (size_t i = 0; i + 192 <= x.size(); i += 192) {
        if (i % 48000 == 0) {
            p.variationSeed += 1;
            p.spectralReshapeDb = p.spectralReshapeDb > 5.0f ? 3.0f : 6.0f;
            e.setParams(p);
        }
        e.process(&x[i], &out[i], 192);
    }
    CHECK(vt::g_allocations.load() - before == 0);
}
