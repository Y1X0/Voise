// JNI surface for com.voiceanon.app.engine.NativeEngine.
// Parameters and statistics cross the boundary as packed float arrays whose
// layouts are mirrored (and documented) in NativeEngine.kt.
#include <jni.h>

#include <algorithm>
#include <string>
#include <vector>

#include "duplex_engine.h"
#include "voiceanon/analysis.h"
#include "voiceanon/presets.h"

using voiceanon::DuplexEngine;
using voiceanon::Params;

namespace {

DuplexEngine& engine() {
    static DuplexEngine e;
    return e;
}

constexpr int kParamCount = 11;
constexpr int kStatsCount = 30;

Params unpackParams(const float* v) {
    Params p;
    p.anonymize = v[0] > 0.5f;
    p.pitchSemitones = v[1];
    p.formantPercent = v[2];
    p.intonation = v[3];
    p.tiltDb = v[4];
    p.clarity = v[5];
    p.noiseSuppression = v[6];
    p.outputGainDb = v[7];
    p.agc = v[8] > 0.5f;
    const int d = static_cast<int>(v[9]);
    p.direction = d > 0 ? voiceanon::Direction::Up : d < 0 ? voiceanon::Direction::Down : voiceanon::Direction::Auto;
    p.autoDirectionHint = static_cast<int>(v[10]);
    return p;
}

void packParams(const Params& p, float* v) {
    v[0] = p.anonymize ? 1.0f : 0.0f;
    v[1] = p.pitchSemitones;
    v[2] = p.formantPercent;
    v[3] = p.intonation;
    v[4] = p.tiltDb;
    v[5] = p.clarity;
    v[6] = p.noiseSuppression;
    v[7] = p.outputGainDb;
    v[8] = p.agc ? 1.0f : 0.0f;
    v[9] = static_cast<float>(static_cast<int>(p.direction));
    v[10] = static_cast<float>(p.autoDirectionHint);
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeStart(JNIEnv*, jobject, jint sampleRate,
                                                                                  jboolean voiceComm, jint inputDevice,
                                                                                  jint outputDevice) {
    voiceanon::DuplexConfig c;
    c.sampleRate = sampleRate;
    c.voiceCommunicationPreset = voiceComm;
    c.inputDeviceId = inputDevice;
    c.outputDeviceId = outputDevice;
    return engine().start(c) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeStop(JNIEnv*, jobject) { engine().stop(); }

JNIEXPORT jboolean JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeIsRunning(JNIEnv*, jobject) {
    return engine().isRunning() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeSetParams(JNIEnv* env, jobject,
                                                                                 jfloatArray values) {
    if (env->GetArrayLength(values) < kParamCount) return;
    float v[kParamCount];
    env->GetFloatArrayRegion(values, 0, kParamCount, v);
    engine().setParams(unpackParams(v));
}

JNIEXPORT jfloatArray JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativePresetParams(JNIEnv* env, jobject,
                                                                                           jfloat strength) {
    float v[kParamCount];
    packParams(voiceanon::paramsForStrength(strength), v);
    jfloatArray out = env->NewFloatArray(kParamCount);
    env->SetFloatArrayRegion(out, 0, kParamCount, v);
    return out;
}

JNIEXPORT jfloatArray JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeGetStats(JNIEnv* env, jobject) {
    float v[kStatsCount] = {0};
    const voiceanon::DuplexStats s = engine().stats();
    voiceanon::Metrics m;
    double algo = 0.0;
    const bool haveMetrics = engine().engineMetrics(&m, &algo);
    v[0] = s.running ? 1.0f : 0.0f;
    v[1] = static_cast<float>(s.sampleRate);
    v[2] = static_cast<float>(algo);
    v[3] = static_cast<float>(s.inputLatencyMs);
    v[4] = static_cast<float>(s.outputLatencyMs);
    v[5] = (s.inputLatencyMs >= 0 && s.outputLatencyMs >= 0)
               ? static_cast<float>(algo + s.inputLatencyMs + s.outputLatencyMs)
               : -1.0f;
    v[6] = s.callbackLoadAvg;
    v[7] = s.callbackLoadMax;
    v[8] = static_cast<float>(s.framesPerCallback);
    v[9] = static_cast<float>(s.inputUnderruns);
    v[10] = static_cast<float>(s.outputXruns);
    if (haveMetrics) {
        v[11] = static_cast<float>(m.lostFrames);
        v[12] = m.f0In;
        v[13] = m.f0OutEstimate;
        v[14] = m.meanF0;
        v[15] = m.voiced ? 1.0f : 0.0f;
        v[16] = m.vad ? 1.0f : 0.0f;
        v[17] = m.pitchRatio;
        v[18] = m.formantRatio;
        v[19] = static_cast<float>(m.direction);
        v[20] = m.inputRmsDb;
        v[21] = m.outputRmsDb;
        v[22] = m.outputPeakDb;
        v[23] = m.limiterGainDb;
        v[24] = m.agcGainDb;
        v[25] = m.noiseFloorDb;
        v[28] = static_cast<float>(m.sanitizedSamples);
    }
    v[26] = engine().captureProgress();
    v[27] = static_cast<float>(s.restarts);
    jfloatArray out = env->NewFloatArray(kStatsCount);
    env->SetFloatArrayRegion(out, 0, kStatsCount, v);
    return out;
}

JNIEXPORT jboolean JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeStartCapture(JNIEnv*, jobject,
                                                                                        jfloat seconds) {
    return engine().startCapture(seconds) ? JNI_TRUE : JNI_FALSE;
}

// Returns [dry, wet, [sampleRate]] or null when no completed capture exists.
JNIEXPORT jobjectArray JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeReadCapture(JNIEnv* env, jobject) {
    std::vector<float> dry, wet;
    int rate = 0;
    const int n = engine().readCapture(dry, wet, &rate);
    if (n <= 0) return nullptr;
    jclass floatArrayClass = env->FindClass("[F");
    jobjectArray result = env->NewObjectArray(3, floatArrayClass, nullptr);
    jfloatArray d = env->NewFloatArray(n), w = env->NewFloatArray(n), r = env->NewFloatArray(1);
    env->SetFloatArrayRegion(d, 0, n, dry.data());
    env->SetFloatArrayRegion(w, 0, n, wet.data());
    const float rf = static_cast<float>(rate);
    env->SetFloatArrayRegion(r, 0, 1, &rf);
    env->SetObjectArrayElement(result, 0, d);
    env->SetObjectArrayElement(result, 1, w);
    env->SetObjectArrayElement(result, 2, r);
    return result;
}

// Renders `dry` through a fresh engine (same streaming code, 192-frame blocks)
// with the given strength / direction; output is latency-aligned with the input.
// Used by the blind listening-test recorder to make B/C/D from one take.
JNIEXPORT jfloatArray JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeRenderOffline(
    JNIEnv* env, jobject, jfloatArray dry, jint sampleRate, jfloat strength, jint direction) {
    const jsize n = env->GetArrayLength(dry);
    std::vector<float> in(n);
    env->GetFloatArrayRegion(dry, 0, n, in.data());
    voiceanon::Engine e(sampleRate);
    Params p = voiceanon::paramsForStrength(strength);
    p.direction = direction > 0 ? voiceanon::Direction::Up
                  : direction < 0 ? voiceanon::Direction::Down : voiceanon::Direction::Auto;
    e.setParams(p);
    e.reset();
    const int lat = e.latencySamples();
    in.resize(static_cast<size_t>(n) + lat + 192, 0.0f);
    std::vector<float> out(in.size(), 0.0f);
    for (size_t i = 0; i < in.size(); i += 192) {
        const int m = static_cast<int>(std::min<size_t>(192, in.size() - i));
        e.process(&in[i], &out[i], m);
    }
    jfloatArray result = env->NewFloatArray(n);
    env->SetFloatArrayRegion(result, 0, n, out.data() + lat);
    return result;
}

// Offline comparison (same code as the host test-suite). Returns JSON.
JNIEXPORT jstring JNICALL Java_com_voiceanon_app_engine_NativeEngine_nativeCompare(JNIEnv* env, jobject,
                                                                                  jfloatArray dry, jfloatArray wet,
                                                                                  jint sampleRate) {
    const jsize n = std::min(env->GetArrayLength(dry), env->GetArrayLength(wet));
    std::vector<float> a(n), b(n);
    env->GetFloatArrayRegion(dry, 0, n, a.data());
    env->GetFloatArrayRegion(wet, 0, n, b.data());
    const voiceanon::analysis::Comparison c = voiceanon::analysis::compare(a.data(), b.data(), n, sampleRate);
    return env->NewStringUTF(voiceanon::analysis::toJson(c).c_str());
}

}  // extern "C"
