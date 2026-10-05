// Full-duplex low-latency audio I/O around the voice anonymization engine.
//
// Microphone (Oboe/AAudio input stream, read non-blocking from inside the
// output callback) -> voiceanon::Engine -> output stream (headphones/speaker).
// The output stream's data callback is the single real-time thread; it never
// locks or allocates.
#pragma once

#include <cstring>  // must precede Oboe.h (FullDuplexStream.h uses memset)

#include <oboe/Oboe.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "voiceanon/engine.h"

namespace voiceanon {

struct DuplexConfig {
    int sampleRate = 48000;
    bool voiceCommunicationPreset = false;  // enables platform AEC/NS, usually higher latency
    int inputDeviceId = 0;                  // 0 = default
    int outputDeviceId = 0;
};

struct DuplexStats {
    bool running = false;
    int sampleRate = 0;
    int framesPerCallback = 0;
    double inputLatencyMs = -1.0;
    double outputLatencyMs = -1.0;
    float callbackLoadAvg = 0.0f;  // processing time / callback period
    float callbackLoadMax = 0.0f;  // max since last query
    long inputUnderruns = 0;       // callbacks where the mic delivered fewer frames than needed
    long outputXruns = 0;
    long restarts = 0;
};

class DuplexEngine : public oboe::AudioStreamDataCallback, public oboe::AudioStreamErrorCallback {
public:
    DuplexEngine();
    ~DuplexEngine() override;

    bool start(const DuplexConfig& config);
    void stop();
    bool isRunning() const { return running_.load(); }

    void setParams(const Params& p);
    Params params() const { return params_; }
    DuplexStats stats();
    bool engineMetrics(Metrics* out, double* algorithmicLatencyMs);

    bool startCapture(double seconds);
    float captureProgress();
    int readCapture(std::vector<float>& dry, std::vector<float>& wet, int* sampleRate);

    oboe::DataCallbackResult onAudioReady(oboe::AudioStream* stream, void* audioData, int32_t numFrames) override;
    void onErrorAfterClose(oboe::AudioStream* stream, oboe::Result error) override;

private:
    bool openAndStartLocked();
    void closeLocked();
    void restartAsync();

    std::mutex lock_;
    DuplexConfig config_;
    Params params_;
    std::shared_ptr<oboe::AudioStream> input_, output_;
    std::unique_ptr<Engine> engine_;
    std::vector<float> inBuf_;
    std::atomic<bool> running_{false};
    std::atomic<bool> restarting_{false};
    int drainCallbacks_ = 0;

    std::atomic<float> loadAvg_{0.0f}, loadMax_{0.0f};
    std::atomic<long> inputUnderruns_{0}, restarts_{0};
    std::atomic<int> framesPerCallback_{0};
};

}  // namespace voiceanon
