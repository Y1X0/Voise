// voiceanon_eval - offline evaluation of the real-time engine on WAV files or on
// the built-in synthetic corpus. Audio is streamed through the engine in small
// blocks exactly as on the device; dry/wet are then latency-aligned and compared.
//
//   voiceanon_eval [options] file.wav [file2.wav ...]
//   voiceanon_eval --synthetic [options]
//   voiceanon_eval --pair a.wav b.wav        (identity-proxy between two dry files)
//
// Options: --preset natural|balanced|strong   --strength 0..1
//          --pitch ST --formant PCT --intonation G --tilt DB --clarity C --noagc
//          --direction auto|up|down           --ns 0..1     --block N
//          --rate HZ (synthetic only)         --out DIR (write processed WAVs)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "synth.h"
#include "voiceanon/analysis.h"
#include "voiceanon/engine.h"
#include "voiceanon/presets.h"
#include "wav.h"

using namespace voiceanon;

namespace {

std::vector<float> runEngine(const std::vector<float>& x, double fs, const Params& p, int block, int* latency) {
    Engine e(fs);
    e.setParams(p);
    e.reset();
    *latency = e.latencySamples();
    // Feed the signal plus enough trailing silence to flush the latency.
    std::vector<float> in = x;
    in.resize(x.size() + *latency + block, 0.0f);
    std::vector<float> out(in.size(), 0.0f);
    for (size_t i = 0; i < in.size(); i += block) {
        const int n = static_cast<int>(std::min<size_t>(block, in.size() - i));
        e.process(&in[i], &out[i], n);
    }
    // Align: wet[i + latency] corresponds to dry[i].
    return std::vector<float>(out.begin() + *latency, out.begin() + *latency + x.size());
}

std::string g_dumpDir;

void dump(const std::string& name, const std::vector<float>& dry, const std::vector<float>& wet, double fs) {
    if (g_dumpDir.empty()) return;
    const int n = static_cast<int>(dry.size());
    const std::vector<float> fd = analysis::trackPitch(dry.data(), n, fs);
    const std::vector<float> fw = analysis::trackPitch(wet.data(), n, fs);
    std::vector<int> cw, cd;
    analysis::countClicks(wet.data(), n, fs, &cw);
    analysis::countClicks(dry.data(), n, fs, &cd);
    FILE* f = std::fopen((g_dumpDir + "/" + name + ".json").c_str(), "w");
    if (!f) return;
    auto arr = [&](const char* key, const auto& v, bool last) {
        std::fprintf(f, "\"%s\":[", key);
        for (size_t i = 0; i < v.size(); ++i) std::fprintf(f, i ? ",%g" : "%g", static_cast<double>(v[i]));
        std::fprintf(f, last ? "]" : "],");
    };
    std::fprintf(f, "{\"fs\":%g,", fs);
    arr("f0Dry", fd, false);
    arr("f0Wet", fw, false);
    arr("clicksWet", cw, false);
    arr("clicksDry", cd, true);
    std::fprintf(f, "}");
    std::fclose(f);
    testing::writeWav(g_dumpDir + "/" + name + "_dry.wav", dry, static_cast<int>(fs));
    testing::writeWav(g_dumpDir + "/" + name + "_wet.wav", wet, static_cast<int>(fs));
}

void report(const std::string& name, const std::vector<float>& dry, const std::vector<float>& wet, double fs,
            int latency) {
    const analysis::Comparison c = analysis::compare(dry.data(), wet.data(), static_cast<int>(dry.size()), fs);
    std::printf("{\"name\":\"%s\",\"fs\":%d,\"latencyMs\":%.2f,\"metrics\":%s}\n", name.c_str(), static_cast<int>(fs),
                1000.0 * latency / fs, analysis::toJson(c).c_str());
    dump(name, dry, wet, fs);
}

std::string baseName(const std::string& p) {
    const size_t s = p.find_last_of('/');
    std::string b = s == std::string::npos ? p : p.substr(s + 1);
    const size_t d = b.find_last_of('.');
    return d == std::string::npos ? b : b.substr(0, d);
}

}  // namespace

int main(int argc, char** argv) {
    Params p = presetParams(Preset::Balanced);
    std::string preset = "balanced", outDir;
    bool synthetic = false, pair = false;
    int block = 192;
    double rate = 48000.0;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--preset") {
            Preset pr;
            preset = next();
            if (!parsePreset(preset, &pr)) {
                std::fprintf(stderr, "unknown preset\n");
                return 2;
            }
            const Params q = presetParams(pr);
            p.pitchSemitones = q.pitchSemitones;
            p.formantPercent = q.formantPercent;
            p.intonation = q.intonation;
            p.tiltDb = q.tiltDb;
        } else if (a == "--strength") {
            const Params q = paramsForStrength(static_cast<float>(std::atof(next().c_str())));
            p.pitchSemitones = q.pitchSemitones;
            p.formantPercent = q.formantPercent;
            p.intonation = q.intonation;
            p.tiltDb = q.tiltDb;
            preset = "custom";
        } else if (a == "--direction") {
            const std::string d = next();
            p.direction = d == "up" ? Direction::Up : d == "down" ? Direction::Down : Direction::Auto;
        } else if (a == "--pitch") {
            p.pitchSemitones = static_cast<float>(std::atof(next().c_str()));
            preset = "custom";
        } else if (a == "--formant") {
            p.formantPercent = static_cast<float>(std::atof(next().c_str()));
            preset = "custom";
        } else if (a == "--intonation") {
            p.intonation = static_cast<float>(std::atof(next().c_str()));
        } else if (a == "--tilt") {
            p.tiltDb = static_cast<float>(std::atof(next().c_str()));
        } else if (a == "--clarity") {
            p.clarity = static_cast<float>(std::atof(next().c_str()));
        } else if (a == "--noagc") {
            p.agc = false;
        } else if (a == "--ns") {
            p.noiseSuppression = static_cast<float>(std::atof(next().c_str()));
        } else if (a == "--block") {
            block = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--rate") {
            rate = std::atof(next().c_str());
        } else if (a == "--out") {
            outDir = next();
        } else if (a == "--dump") {
            g_dumpDir = next();
        } else if (a == "--synthetic") {
            synthetic = true;
        } else if (a == "--pair") {
            pair = true;
        } else {
            files.push_back(a);
        }
    }

    if (pair) {
        if (files.size() != 2) return 2;
        std::vector<float> a, b;
        int fa = 0, fb = 0;
        if (!testing::readWav(files[0], a, fa) || !testing::readWav(files[1], b, fb) || fa != fb) return 1;
        const int n = static_cast<int>(std::min(a.size(), b.size()));
        const double cos = analysis::cosineSimilarity(analysis::meanMfcc(a.data(), static_cast<int>(a.size()), fa),
                                                      analysis::meanMfcc(b.data(), static_cast<int>(b.size()), fb));
        const double fr = analysis::estimateFormantRatio(a.data(), b.data(), n, fa);
        const float f0a = analysis::medianVoicedF0(analysis::trackPitch(a.data(), static_cast<int>(a.size()), fa));
        const float f0b = analysis::medianVoicedF0(analysis::trackPitch(b.data(), static_cast<int>(b.size()), fb));
        std::printf("{\"pair\":[\"%s\",\"%s\"],\"mfccCosine\":%.3f,\"formantRatio\":%.3f,\"f0a\":%.1f,\"f0b\":%.1f}\n",
                    baseName(files[0]).c_str(), baseName(files[1]).c_str(), cos, fr, f0a, f0b);
        return 0;
    }

    if (synthetic) {
        struct Item {
            std::string name;
            testing::VoiceProfile v;
            std::string text;
        };
        const Item items[] = {{"male_arabic", testing::maleVoice(), testing::arabicText()},
                              {"female_arabic", testing::femaleVoice(), testing::arabicText()},
                              {"male_english", testing::maleVoice(), testing::englishText()},
                              {"female_english", testing::femaleVoice(), testing::englishText()}};
        uint32_t seed = 1;
        for (const Item& it : items) {
            const std::vector<float> dry =
                testing::pad(testing::synthesizeSpeech(it.text, it.v, rate, seed++, -20.0), rate, 0.3, 0.3);
            int latency = 0;
            const std::vector<float> wet = runEngine(dry, rate, p, block, &latency);
            report("synthetic_" + it.name + "_" + preset, dry, wet, rate, latency);
            if (!outDir.empty()) {
                testing::writeWav(outDir + "/synthetic_" + it.name + "_dry.wav", dry, static_cast<int>(rate));
                testing::writeWav(outDir + "/synthetic_" + it.name + "_" + preset + ".wav", wet, static_cast<int>(rate));
            }
        }
        return 0;
    }

    for (const std::string& f : files) {
        std::vector<float> dry;
        int fs = 0;
        if (!testing::readWav(f, dry, fs)) {
            std::fprintf(stderr, "cannot read %s\n", f.c_str());
            continue;
        }
        int latency = 0;
        const std::vector<float> wet = runEngine(dry, fs, p, block, &latency);
        report(baseName(f) + "_" + preset, dry, wet, fs, latency);
        if (!outDir.empty()) testing::writeWav(outDir + "/" + baseName(f) + "_" + preset + ".wav", wet, fs);
    }
    return 0;
}
