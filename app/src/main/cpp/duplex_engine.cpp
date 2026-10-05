#include "duplex_engine.h"

#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#define LOG_TAG "VoiceAnonNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace voiceanon {
namespace {
constexpr int kInputBufferFrames = 8192;
constexpr int kDrainCallbacks = 20;
}  // namespace

DuplexEngine::DuplexEngine() { inBuf_.assign(kInputBufferFrames, 0.0f); }

DuplexEngine::~DuplexEngine() { stop(); }

bool DuplexEngine::start(const DuplexConfig& config) {
    std::lock_guard<std::mutex> g(lock_);
    if (running_.load()) return true;
    config_ = config;
    return openAndStartLocked();
}

void DuplexEngine::stop() {
    std::lock_guard<std::mutex> g(lock_);
    closeLocked();
}

bool DuplexEngine::openAndStartLocked() {
    // --- Input (microphone) ---------------------------------------------------
    oboe::AudioStreamBuilder in;
    in.setDirection(oboe::Direction::Input)
        ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
        ->setSharingMode(oboe::SharingMode::Exclusive)
        ->setFormat(oboe::AudioFormat::Float)
        ->setChannelCount(oboe::ChannelCount::Mono)
        ->setSampleRate(config_.sampleRate)
        ->setSampleRateConversionQuality(oboe::SampleRateConversionQuality::Medium)
        ->setInputPreset(config_.voiceCommunicationPreset ? oboe::InputPreset::VoiceCommunication
                                                          : oboe::InputPreset::VoiceRecognition)
        ->setErrorCallback(this);
    if (config_.inputDeviceId > 0) in.setDeviceId(config_.inputDeviceId);
    oboe::Result r = in.openStream(input_);
    if (r != oboe::Result::OK) {
        LOGW("open input failed: %s", oboe::convertToText(r));
        closeLocked();
        return false;
    }
    const int rate = input_->getSampleRate();

    // --- Output (headphones / speaker) ----------------------------------------
    oboe::AudioStreamBuilder out;
    out.setDirection(oboe::Direction::Output)
        ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
        ->setSharingMode(oboe::SharingMode::Exclusive)
        ->setFormat(oboe::AudioFormat::Float)
        ->setChannelCount(oboe::ChannelCount::Mono)
        ->setSampleRate(rate)
        ->setSampleRateConversionQuality(oboe::SampleRateConversionQuality::Medium)
        ->setUsage(oboe::Usage::Media)
        ->setContentType(oboe::ContentType::Speech)
        ->setDataCallback(this)
        ->setErrorCallback(this);
    if (config_.outputDeviceId > 0) out.setDeviceId(config_.outputDeviceId);
    r = out.openStream(output_);
    if (r != oboe::Result::OK) {
        LOGW("open output failed: %s", oboe::convertToText(r));
        closeLocked();
        return false;
    }
    // Two bursts of output buffering: low latency with some glitch protection.
    output_->setBufferSizeInFrames(output_->getFramesPerBurst() * 2);

    engine_ = std::make_unique<Engine>(static_cast<double>(rate), kInputBufferFrames);
    engine_->setParams(params_);
    engine_->reset();
    drainCallbacks_ = kDrainCallbacks;
    loadAvg_.store(0.0f);
    loadMax_.store(0.0f);

    r = input_->requestStart();
    if (r == oboe::Result::OK) r = output_->requestStart();
    if (r != oboe::Result::OK) {
        LOGW("start failed: %s", oboe::convertToText(r));
        closeLocked();
        return false;
    }
    running_.store(true);
    LOGI("duplex started: %d Hz, input sharing %d, output burst %d", rate,
         static_cast<int>(input_->getSharingMode()), output_->getFramesPerBurst());
    return true;
}

void DuplexEngine::closeLocked() {
    running_.store(false);
    if (output_) {
        output_->requestStop();
        output_->close();
        output_.reset();
    }
    if (input_) {
        input_->requestStop();
        input_->close();
        input_.reset();
    }
    // engine_ is kept so metrics / captures remain readable after stopping.
}

void DuplexEngine::setParams(const Params& p) {
    params_ = p;
    std::lock_guard<std::mutex> g(lock_);
    if (engine_) engine_->setParams(p);
}

oboe::DataCallbackResult DuplexEngine::onAudioReady(oboe::AudioStream* stream, void* audioData, int32_t numFrames) {
    const auto t0 = std::chrono::steady_clock::now();
    float* out = static_cast<float*>(audioData);
    oboe::AudioStream* in = input_.get();
    Engine* engine = engine_.get();
    if (in == nullptr || engine == nullptr) {
        std::memset(out, 0, sizeof(float) * numFrames);
        return oboe::DataCallbackResult::Continue;
    }
    framesPerCallback_.store(numFrames, std::memory_order_relaxed);

    if (drainCallbacks_ > 0) {
        // Drain whatever the mic buffered while the streams were starting so
        // the steady-state input->output delay stays minimal.
        int drained = 0;
        for (int i = 0; i < 16; ++i) {
            auto res = in->read(inBuf_.data(), std::min<int32_t>(numFrames, kInputBufferFrames), 0);
            if (!res || res.value() <= 0) break;
            drained += res.value();
        }
        if (drained > 0) --drainCallbacks_;
        std::memset(out, 0, sizeof(float) * numFrames);
        return oboe::DataCallbackResult::Continue;
    }

    int done = 0;
    while (done < numFrames) {
        const int want = std::min<int32_t>(numFrames - done, kInputBufferFrames);
        auto res = in->read(inBuf_.data(), want, 0);
        const int got = res ? std::max<int32_t>(0, res.value()) : 0;
        if (got > 0) engine->process(inBuf_.data(), out + done, got);
        if (got < want) {
            // Mic underrun / lost frames: the engine conceals them smoothly.
            inputUnderruns_.fetch_add(1, std::memory_order_relaxed);
            engine->process(nullptr, out + done + got, want - got);
        }
        done += want;
    }

    const double cost = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double period = static_cast<double>(numFrames) / stream->getSampleRate();
    const float load = static_cast<float>(cost / period);
    loadAvg_.store(loadAvg_.load(std::memory_order_relaxed) * 0.98f + load * 0.02f, std::memory_order_relaxed);
    if (load > loadMax_.load(std::memory_order_relaxed)) loadMax_.store(load, std::memory_order_relaxed);
    return oboe::DataCallbackResult::Continue;
}

void DuplexEngine::onErrorAfterClose(oboe::AudioStream* stream, oboe::Result error) {
    LOGW("stream error %s - restarting", oboe::convertToText(error));
    if (error == oboe::Result::ErrorDisconnected) restartAsync();
}

void DuplexEngine::restartAsync() {
    if (restarting_.exchange(true)) return;
    std::thread([this] {
        {
            std::lock_guard<std::mutex> g(lock_);
            const bool wasRunning = running_.load() || output_ != nullptr || input_ != nullptr;
            closeLocked();
            if (wasRunning) {
                if (openAndStartLocked()) restarts_.fetch_add(1);
            }
        }
        restarting_.store(false);
    }).detach();
}

DuplexStats DuplexEngine::stats() {
    DuplexStats s;
    std::lock_guard<std::mutex> g(lock_);
    s.running = running_.load();
    s.framesPerCallback = framesPerCallback_.load();
    s.callbackLoadAvg = loadAvg_.load();
    s.callbackLoadMax = loadMax_.exchange(0.0f);
    s.inputUnderruns = inputUnderruns_.load();
    s.restarts = restarts_.load();
    if (output_) {
        s.sampleRate = output_->getSampleRate();
        auto l = output_->calculateLatencyMillis();
        if (l) s.outputLatencyMs = l.value();
        auto x = output_->getXRunCount();
        if (x) s.outputXruns = x.value();
    }
    if (input_) {
        auto l = input_->calculateLatencyMillis();
        if (l) s.inputLatencyMs = l.value();
    }
    return s;
}

bool DuplexEngine::engineMetrics(Metrics* out, double* algorithmicLatencyMs) {
    std::lock_guard<std::mutex> g(lock_);
    if (!engine_) return false;
    *out = engine_->metrics();
    *algorithmicLatencyMs = engine_->latencyMs();
    return true;
}

bool DuplexEngine::startCapture(double seconds) {
    std::lock_guard<std::mutex> g(lock_);
    return engine_ && running_.load() && engine_->startCapture(seconds);
}

float DuplexEngine::captureProgress() {
    std::lock_guard<std::mutex> g(lock_);
    return engine_ ? engine_->captureProgress() : 0.0f;
}

int DuplexEngine::readCapture(std::vector<float>& dry, std::vector<float>& wet, int* sampleRate) {
    std::lock_guard<std::mutex> g(lock_);
    if (!engine_) return 0;
    *sampleRate = static_cast<int>(engine_->sampleRate());
    return engine_->readCapture(dry, wet);
}

}  // namespace voiceanon
