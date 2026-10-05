// Offline measurement utilities shared by the host test-suite, the WAV
// evaluation tool and the on-device Test Mode (via JNI).
//
// IMPORTANT: the similarity numbers produced here are *proxies* (MFCC / LTAS
// statistics, F0 and formant statistics). They are NOT a speaker-verification
// system and do not prove anonymity against an ASV model or a human expert.
#pragma once

#include <string>
#include <vector>

namespace voiceanon {
namespace analysis {

// Frame-wise F0 (Hz, 0 = unvoiced) with a 10 ms hop.
std::vector<float> trackPitch(const float* x, int n, double fs);
float medianVoicedF0(const std::vector<float>& f0);

// Long-term average LPC envelope (dB) over voiced/active frames, sampled on a
// linear grid of `points` frequencies between 0 and maxHz.
std::vector<float> averageEnvelopeDb(const float* x, int n, double fs, double maxHz, int points);

// Estimated spectral-envelope (formant) scaling factor between a and b:
// envelope_b(f) ~= envelope_a(f / ratio). Searched in [0.7, 1.4]; returns -1
// when the best fit lies on the search boundary (unreliable estimate).
double estimateFormantRatio(const float* a, const float* b, int n, double fs);

// Mean MFCC vector (c1..c12) over active frames.
std::vector<float> meanMfcc(const float* x, int n, double fs);
double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

struct Comparison {
    // Pitch
    double f0DryHz = 0, f0WetHz = 0;
    double pitchShiftSemitones = 0;  // median frame-wise shift (frames voiced in both)
    double intonationCorrelation = 0;  // corr. of log-F0 contours (1 = melody preserved)
    double f0GrossErrorRate = 0;       // frames whose shift deviates > 3 st from the median
    double voicingAgreement = 0;       // fraction of dry-voiced frames also voiced in wet
    // Timbre / identity proxies
    double formantRatio = 1;     // -1 = could not be estimated reliably
    double mfccCosine = 1;       // cosine similarity of mean MFCC vectors (1 = identical)
    double ltasDistanceDb = 0;   // RMS difference of mean-removed long-term envelopes
    // Content / continuity / artefacts
    double envelopeCorrelation = 0;  // broadband syllabic envelope correlation
    int dropouts = 0;  // 10 ms speech frames (>100 Hz) that are >25 dB weaker in wet, relative to each signal's level
    int clippedSamples = 0;          // |wet| >= 0.999
    int newClicks = 0;               // impulsive discontinuities present in wet only
    double wetPeak = 0, dryRmsDb = -120, wetRmsDb = -120;
};

// dry and wet must be time-aligned (the engine's capture and the tools do this).
Comparison compare(const float* dry, const float* wet, int n, double fs);

std::string toJson(const Comparison& c);

// Count impulsive clicks: second-difference spikes far above the local level.
// relaxed = true lowers the thresholds (used to find transients in the dry
// reference so that natural bursts are not reported as processing clicks).
int countClicks(const float* x, int n, double fs, std::vector<int>* positions = nullptr, bool relaxed = false);

double rmsDb(const float* x, int n);

}  // namespace analysis
}  // namespace voiceanon
