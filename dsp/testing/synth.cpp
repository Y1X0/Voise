#include "synth.h"

#include <cmath>
#include <map>
#include <random>
#include <sstream>

#include "voiceanon/biquad.h"

namespace voiceanon {
namespace testing {
namespace {

enum class Kind { Vowel, Sonorant, Pharyngeal, Aspirate, Fricative, VoicedFricative, Stop, VoicedStop, Pause };

struct Phone {
    Kind kind;
    double dur;
    double f[4];
    double voice;   // voicing amplitude
    double asp;     // noise through formants
    double fric;    // band-pass noise amplitude
    double fricHz;
    double fricQ;
};

const std::map<std::string, Phone>& table() {
    static const std::map<std::string, Phone> t = {
        {"a", {Kind::Vowel, 0.11, {730, 1090, 2440, 3400}, 1.0, 0, 0, 0, 1}},
        {"aa", {Kind::Vowel, 0.20, {730, 1090, 2440, 3400}, 1.0, 0, 0, 0, 1}},
        {"i", {Kind::Vowel, 0.09, {300, 2250, 3000, 3600}, 0.9, 0, 0, 0, 1}},
        {"ii", {Kind::Vowel, 0.18, {300, 2250, 3000, 3600}, 0.9, 0, 0, 0, 1}},
        {"u", {Kind::Vowel, 0.09, {320, 870, 2240, 3300}, 0.9, 0, 0, 0, 1}},
        {"uu", {Kind::Vowel, 0.18, {320, 870, 2240, 3300}, 0.9, 0, 0, 0, 1}},
        {"e", {Kind::Vowel, 0.10, {500, 1850, 2500, 3500}, 1.0, 0, 0, 0, 1}},
        {"o", {Kind::Vowel, 0.11, {550, 850, 2450, 3400}, 1.0, 0, 0, 0, 1}},
        {"ae", {Kind::Vowel, 0.11, {660, 1700, 2400, 3400}, 1.0, 0, 0, 0, 1}},
        {"@", {Kind::Vowel, 0.06, {500, 1500, 2500, 3500}, 0.8, 0, 0, 0, 1}},
        {"er", {Kind::Vowel, 0.12, {490, 1350, 1690, 3300}, 0.9, 0, 0, 0, 1}},
        {"w", {Kind::Sonorant, 0.05, {320, 650, 2200, 3300}, 0.6, 0, 0, 0, 1}},
        {"y", {Kind::Sonorant, 0.05, {280, 2150, 3000, 3600}, 0.6, 0, 0, 0, 1}},
        {"l", {Kind::Sonorant, 0.06, {380, 1300, 2700, 3400}, 0.55, 0, 0, 0, 1}},
        {"r", {Kind::Sonorant, 0.06, {420, 1250, 1600, 3300}, 0.5, 0, 0, 0, 1}},
        {"m", {Kind::Sonorant, 0.07, {280, 1100, 2200, 3300}, 0.25, 0, 0, 0, 1}},
        {"n", {Kind::Sonorant, 0.06, {280, 1700, 2600, 3400}, 0.25, 0, 0, 0, 1}},
        {"3", {Kind::Pharyngeal, 0.08, {750, 1200, 2500, 3400}, 0.55, 0.05, 0, 0, 1}},
        {"H", {Kind::Aspirate, 0.09, {850, 1250, 2500, 3400}, 0.0, 0.20, 0, 0, 1}},
        {"h", {Kind::Aspirate, 0.06, {500, 1500, 2500, 3500}, 0.0, 0.10, 0, 0, 1}},
        {"x", {Kind::Fricative, 0.09, {500, 1500, 2500, 3500}, 0.0, 0, 0.15, 1600, 1.5}},
        {"s", {Kind::Fricative, 0.10, {500, 1500, 2500, 3500}, 0.0, 0, 0.12, 6000, 2.0}},
        {"sh", {Kind::Fricative, 0.10, {500, 1500, 2500, 3500}, 0.0, 0, 0.15, 3200, 1.5}},
        {"f", {Kind::Fricative, 0.09, {500, 1500, 2500, 3500}, 0.0, 0, 0.05, 4000, 0.7}},
        {"z", {Kind::VoicedFricative, 0.08, {300, 1500, 2500, 3500}, 0.2, 0, 0.08, 6000, 2.0}},
        {"p", {Kind::Stop, 0.08, {500, 1500, 2500, 3500}, 0.0, 0, 0.25, 900, 1.0}},
        {"t", {Kind::Stop, 0.08, {500, 1700, 2600, 3500}, 0.0, 0, 0.22, 4500, 1.2}},
        {"k", {Kind::Stop, 0.08, {500, 1500, 2500, 3500}, 0.0, 0, 0.22, 2200, 1.5}},
        {"q", {Kind::Stop, 0.09, {600, 1100, 2500, 3500}, 0.0, 0, 0.22, 1300, 1.5}},
        {"b", {Kind::VoicedStop, 0.07, {300, 1100, 2300, 3400}, 0.08, 0, 0.12, 900, 1.0}},
        {"d", {Kind::VoicedStop, 0.07, {300, 1700, 2600, 3400}, 0.08, 0, 0.12, 4000, 1.2}},
        {"g", {Kind::VoicedStop, 0.07, {300, 1500, 2400, 3400}, 0.08, 0, 0.12, 2200, 1.5}},
        {"_", {Kind::Pause, 0.15, {500, 1500, 2500, 3500}, 0.0, 0, 0, 0, 1}},
        {"|", {Kind::Pause, 0.06, {500, 1500, 2500, 3500}, 0.0, 0, 0, 0, 1}},
    };
    return t;
}

struct Segment {
    int start, end;
    double f[4];
    double voice, asp, fric, fricHz, fricQ;
    double accent;
    bool burst;  // stop: closure then burst at the end
};

struct Resonator {
    double a = 1, b = 0, c = 0, y1 = 0, y2 = 0;
    void set(double fs, double f, double bw) {
        c = -std::exp(-2.0 * M_PI * bw / fs);
        b = 2.0 * std::exp(-M_PI * bw / fs) * std::cos(2.0 * M_PI * f / fs);
        a = 1.0 - b - c;
    }
    double process(double x) {
        const double y = a * x + b * y1 + c * y2;
        y2 = y1;
        y1 = y;
        return y;
    }
};

double rosenberg(double phase) {
    const double tp = 0.40, tn = 0.16;
    if (phase < tp) return 0.5 * (1.0 - std::cos(M_PI * phase / tp));
    if (phase < tp + tn) return std::cos(M_PI * (phase - tp) / (2.0 * tn));
    return 0.0;
}

}  // namespace

VoiceProfile maleVoice() { return {"male", 115.0, 1.0, 0.02, 0.006, 0.04}; }
VoiceProfile femaleVoice() { return {"female", 210.0, 1.17, 0.05, 0.006, 0.04}; }

std::string arabicText() {
    return "| m a r H a b a | k a y f a k _ aa n a | b i x a y r | sh u k r aa n _ 3 a l aa | q a l b ii |";
}

std::string englishText() {
    return "| h e l o w | h a w | aa r | y uu | t @ d e y _ sh ii | s e l z | f i sh _ w e r | i z | i t |";
}

std::vector<float> synthesizeSpeech(const std::string& phonemes, const VoiceProfile& v, double fs, uint32_t seed,
                                    double rmsDbfs) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    // Build segments.
    std::vector<Segment> segs;
    std::istringstream ss(phonemes);
    std::string tok;
    int pos = 0, vowelIndex = 0;
    while (ss >> tok) {
        auto it = table().find(tok);
        if (it == table().end()) continue;
        const Phone& p = it->second;
        Segment s{};
        const double durJitter = 1.0 + 0.1 * uni(rng);
        const int len = static_cast<int>(p.dur * durJitter * fs);
        s.start = pos;
        s.end = pos + len;
        for (int k = 0; k < 4; ++k) s.f[k] = p.f[k] * v.formantScale;
        s.voice = p.voice;
        s.asp = p.asp;
        s.fric = p.fric;
        s.fricHz = std::min(p.fricHz, 0.42 * fs);
        s.fricQ = p.fricQ;
        s.burst = p.kind == Kind::Stop || p.kind == Kind::VoicedStop;
        if (p.kind == Kind::Vowel) s.accent = (vowelIndex++ % 3 == 1) ? 2.5 : 0.0;
        segs.push_back(s);
        pos = s.end;
    }
    const int total = pos;
    std::vector<float> out(total, 0.0f);
    if (total == 0) return out;

    Resonator res[4];
    Resonator asp[4];
    Biquad fricFilter = Biquad::bandpass(fs, 3000, 1.0);
    double sf[4] = {500, 1500, 2500, 3500};
    double sVoice = 0, sAsp = 0, sFric = 0, sAccent = 0;
    const double cForm = 1.0 - std::exp(-1.0 / (0.012 * fs));
    const double cAmp = 1.0 - std::exp(-1.0 / (0.006 * fs));
    const double cFast = 1.0 - std::exp(-1.0 / (0.0015 * fs));
    const double cAccent = 1.0 - std::exp(-1.0 / (0.04 * fs));
    double phase = 0.0, periodScale = 1.0, ampScale = 1.0, prevG = 0.0;
    double lastFricHz = -1, lastFricQ = -1;
    size_t si = 0;
    for (int n = 0; n < total; ++n) {
        while (si + 1 < segs.size() && n >= segs[si].end) ++si;
        const Segment& s = segs[si];
        const double t = static_cast<double>(n) / total;

        double targetVoice = s.voice, targetFric = s.fric, targetAsp = s.asp;
        bool fast = false;
        if (s.burst) {
            // closure (voice bar for voiced stops), then 12 ms burst + short aspiration
            const int burstStart = s.end - static_cast<int>(0.025 * fs);
            if (n < burstStart) {
                targetFric = 0.0;
                targetAsp = 0.0;
            } else {
                fast = true;
                targetVoice = 0.0;
                targetAsp = (n > burstStart + static_cast<int>(0.012 * fs)) ? 0.06 : 0.0;
            }
        }
        for (int k = 0; k < 4; ++k) sf[k] += cForm * (s.f[k] - sf[k]);
        sVoice += cAmp * (targetVoice - sVoice);
        sAsp += cAmp * (targetAsp - sAsp);
        sFric += (fast ? cFast : cAmp) * (targetFric - sFric);
        sAccent += cAccent * (s.accent - sAccent);

        if ((n & 31) == 0) {
            const double bwScale = v.formantScale > 1.05 ? 1.15 : 1.0;
            const double bw[4] = {60 * bwScale, 90 * bwScale, 120 * bwScale, 160 * bwScale};
            for (int k = 0; k < 4; ++k) {
                const double f = std::min(sf[k], 0.45 * fs);
                res[k].set(fs, f, bw[k]);
                asp[k].set(fs, f, bw[k] * 1.5);
            }
            if (s.fricHz != lastFricHz || s.fricQ != lastFricQ) {
                if (s.fricHz > 0) fricFilter.setCoefs(Biquad::bandpass(fs, s.fricHz, s.fricQ));
                lastFricHz = s.fricHz;
                lastFricQ = s.fricQ;
            }
        }

        // Glottal source with declination + accents, jitter and shimmer.
        const double semis = 2.0 - 4.0 * t + sAccent;
        const double f0 = v.f0Hz * std::pow(2.0, semis / 12.0);
        phase += f0 / fs * periodScale;
        if (phase >= 1.0) {
            phase -= 1.0;
            periodScale = 1.0 + v.jitter * gauss(rng);
            ampScale = 1.0 + v.shimmer * gauss(rng);
        }
        const double g = rosenberg(phase);
        const double dg = (g - prevG) * fs / (f0 * 4.0);  // lip radiation (differentiated flow)
        prevG = g;
        const double noise = uni(rng);
        double voiced = sVoice * ampScale * dg + sVoice * v.breathiness * noise * g;
        for (int k = 0; k < 4; ++k) voiced = res[k].process(voiced);
        double aspN = sAsp * noise;
        for (int k = 0; k < 4; ++k) aspN = asp[k].process(aspN);
        const double fricN = sFric * fricFilter.process(static_cast<float>(uni(rng))) * 4.0;
        out[n] = static_cast<float>(voiced + 3.0 * aspN + fricN);
    }

    double e = 0.0;
    for (float x : out) e += static_cast<double>(x) * x;
    const double rms = std::sqrt(e / total);
    const double gain = rms > 0 ? std::pow(10.0, rmsDbfs / 20.0) / rms : 0.0;
    for (float& x : out) x = static_cast<float>(x * gain);
    return out;
}

std::vector<float> whiteNoise(int n, double rmsDbfs, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    const double a = std::pow(10.0, rmsDbfs / 20.0);
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(a * g(rng));
    return x;
}

std::vector<float> pinkNoise(int n, double rmsDbfs, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    // Paul Kellet's refined pink filter.
    double b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    std::vector<float> x(n);
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        const double w = g(rng);
        b0 = 0.99886 * b0 + w * 0.0555179;
        b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.96900 * b2 + w * 0.1538520;
        b3 = 0.86650 * b3 + w * 0.3104856;
        b4 = 0.55000 * b4 + w * 0.5329522;
        b5 = -0.7616 * b5 - w * 0.0168980;
        const double p = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
        b6 = w * 0.115926;
        x[i] = static_cast<float>(p);
        e += p * p;
    }
    const double gain = std::pow(10.0, rmsDbfs / 20.0) / std::sqrt(e / std::max(1, n) + 1e-30);
    for (float& v : x) v = static_cast<float>(v * gain);
    return x;
}

std::vector<float> sine(int n, double fs, double hz, double amplitude) {
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * hz * i / fs));
    return x;
}

void mixAtSnr(std::vector<float>& speech, const std::vector<float>& noise, double snrDb) {
    if (noise.empty()) return;
    double es = 0, en = 0;
    for (size_t i = 0; i < speech.size(); ++i) {
        const float nv = noise[i % noise.size()];
        es += static_cast<double>(speech[i]) * speech[i];
        en += static_cast<double>(nv) * nv;
    }
    const double g = std::sqrt(es / (en + 1e-30)) * std::pow(10.0, -snrDb / 20.0);
    for (size_t i = 0; i < speech.size(); ++i) speech[i] += static_cast<float>(g * noise[i % noise.size()]);
}

std::vector<float> pad(const std::vector<float>& x, double fs, double before, double after) {
    std::vector<float> y(static_cast<size_t>(before * fs), 0.0f);
    y.insert(y.end(), x.begin(), x.end());
    y.resize(y.size() + static_cast<size_t>(after * fs), 0.0f);
    return y;
}

}  // namespace testing
}  // namespace voiceanon
