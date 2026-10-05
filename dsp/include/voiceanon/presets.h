// User-facing presets. The UI exposes three presets plus one "Anonymization
// strength" slider; DSP details (semitones, formant %, intonation, tilt) are
// derived from the strength so users never have to understand them.
#pragma once

#include <string>

#include "voiceanon/engine.h"

namespace voiceanon {

enum class Preset { Natural = 0, Balanced = 1, Strong = 2 };

// Strength 0..1 -> anonymization parameters (other fields keep their defaults).
// The ranges are deliberately bounded so the result never approaches the
// "child" (large upward formant + pitch shift) or "monster" (large downward
// shift) regions: |pitch| <= 4.5 semitones, formant shift <= 16 %.
Params paramsForStrength(float strength);

float presetStrength(Preset p);
Params presetParams(Preset p);
const char* presetName(Preset p);
bool parsePreset(const std::string& name, Preset* out);

}  // namespace voiceanon
