// Deterministic source-filter speech synthesizer used to generate reproducible
// test material (male/female voices, Arabic-like and English-like phoneme
// sequences, quiet/loud levels, noise). It is a Klatt-style cascade formant
// synthesizer: Rosenberg glottal pulses with jitter/shimmer and declining F0,
// four time-varying formant resonators, band-pass noise for fricatives and
// bursts for stops.
//
// This is NOT real human speech. Results on synthetic speech are labelled as
// such in the documentation; real-recording evaluation is done separately with
// `voiceanon_eval` on WAV files.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace voiceanon {
namespace testing {

struct VoiceProfile {
    std::string name;
    double f0Hz;
    double formantScale;
    double breathiness;
    double jitter;
    double shimmer;
};

VoiceProfile maleVoice();
VoiceProfile femaleVoice();

// Phoneme string, space separated. Symbols: vowels a aa i ii u uu e o ae @ er,
// glides/liquids/nasals w y l r m n, pharyngeal 3 (voiced, Arabic ain) and
// H (voiceless, Arabic haa), fricatives h s sh f z x, stops p t k q b d g,
// pauses _ (150 ms) and | (60 ms).
std::vector<float> synthesizeSpeech(const std::string& phonemes, const VoiceProfile& voice, double fs,
                                    uint32_t seed, double rmsDbfs = -20.0);

// Multi-sentence material.
std::string arabicText();   // "marhaba kayfak / ana bikhayr shukran / 3ala qalbi"
std::string englishText();  // "hello how are you today / she sells fish"

std::vector<float> whiteNoise(int n, double rmsDbfs, uint32_t seed);
std::vector<float> pinkNoise(int n, double rmsDbfs, uint32_t seed);
std::vector<float> sine(int n, double fs, double hz, double amplitude);

// Adds noise (looped/truncated) to speech so that speech-RMS / noise-RMS = snrDb.
void mixAtSnr(std::vector<float>& speech, const std::vector<float>& noise, double snrDb);

// Inserts `seconds` of silence before and after.
std::vector<float> pad(const std::vector<float>& x, double fs, double before, double after);

}  // namespace testing
}  // namespace voiceanon
