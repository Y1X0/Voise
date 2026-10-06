// C ABI over the app's UNCHANGED dsp/ front-end blocks, so training, evaluation and the
// Android runtime use bit-identical pitch tracking and noise suppression.
// Built by training/native/voiceanon_native.py (g++ -shared); nothing in dsp/ is modified.
#include <algorithm>
#include <vector>

#include "voiceanon/noise_suppressor.h"
#include "voiceanon/pitch_tracker.h"

extern "C" {

// YIN F0 per hop: pushes `hop` samples at a time and reads the tracker state after each
// push (exactly what the runtime does per 10 ms). Returns the number of frames written.
int va_yin_track(const float* x, int n, double sr, int hop, float* f0, float* confidence) {
    voiceanon::PitchTracker pt(sr);
    const int frames = n / hop;
    for (int t = 0; t < frames; ++t) {
        pt.push(x + static_cast<long>(t) * hop, hop);
        f0[t] = pt.f0();
        if (confidence) confidence[t] = pt.voiced() ? pt.confidence() : 0.0f;
    }
    return frames;
}

// Noise suppressor over a whole signal, hop by hop. Output has the block's fixed latency
// (returned); trailing samples that do not fill a hop are copied through.
int va_noise_suppress(const float* x, int n, double sr, float amount, float* out) {
    voiceanon::NoiseSuppressor ns(sr);
    ns.setAmount(amount);
    const int hop = ns.hop();
    std::vector<float> buf(hop);
    int i = 0;
    for (; i + hop <= n; i += hop) {
        std::copy(x + i, x + i + hop, buf.begin());
        ns.processHop(buf.data(), out + i);
    }
    for (; i < n; ++i) out[i] = x[i];
    return ns.latency();
}

int va_noise_suppressor_hop(double sr) { return voiceanon::NoiseSuppressor(sr).hop(); }

}  // extern "C"
