#include "voiceanon/presets.h"

#include <algorithm>

namespace voiceanon {
namespace {
float lerp(float a, float b, float t) { return a + (b - a) * t; }
}  // namespace

Params paramsForStrength(float strength) {
    const float s = std::min(std::max(strength, 0.0f), 1.0f);
    Params p;
    p.pitchSemitones = lerp(1.0f, 4.5f, s);
    p.formantPercent = lerp(4.0f, 16.0f, s);
    p.intonation = lerp(1.0f, 0.75f, s);
    p.tiltDb = lerp(0.5f, 3.0f, s);
    p.clarity = 0.4f;
    p.noiseSuppression = 0.5f;
    p.outputGainDb = 0.0f;
    p.agc = true;
    p.direction = Direction::Auto;
    return p;
}

float presetStrength(Preset p) {
    switch (p) {
        case Preset::Natural: return 0.2f;
        case Preset::Balanced: return 0.55f;
        case Preset::Strong: return 0.9f;
    }
    return 0.55f;
}

Params presetParams(Preset p) { return paramsForStrength(presetStrength(p)); }

const char* presetName(Preset p) {
    switch (p) {
        case Preset::Natural: return "natural";
        case Preset::Balanced: return "balanced";
        case Preset::Strong: return "strong";
    }
    return "balanced";
}

bool parsePreset(const std::string& name, Preset* out) {
    if (name == "natural" || name == "subtle") *out = Preset::Natural;
    else if (name == "balanced") *out = Preset::Balanced;
    else if (name == "strong") *out = Preset::Strong;
    else return false;
    return true;
}

}  // namespace voiceanon
