// voiceanon_bench - per-callback processing-time benchmark of the engine.
// Simulates audio callbacks of typical Android burst sizes and reports the
// real-time factor and the worst-case callback cost relative to its deadline.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "synth.h"
#include "voiceanon/engine.h"
#include "voiceanon/presets.h"

using namespace voiceanon;

int main() {
    const double rates[] = {16000.0, 48000.0};
    const int blocks[] = {96, 192, 480};
    std::printf("| rate | block | callback deadline | mean cost | p99 cost | max cost | real-time factor |\n");
    std::printf("|---|---|---|---|---|---|---|\n");
    for (double fs : rates) {
        const std::vector<float> speech = testing::synthesizeSpeech(
            testing::arabicText() + " _ " + testing::englishText(), testing::maleVoice(), fs, 1, -20.0);
        std::vector<float> noise = testing::pinkNoise(static_cast<int>(speech.size()), -45, 2);
        std::vector<float> x(speech.size());
        for (size_t i = 0; i < x.size(); ++i) x[i] = speech[i] + noise[i];
        for (int block : blocks) {
            Engine e(fs);
            e.setParams(presetParams(Preset::Strong));
            e.reset();
            std::vector<float> out(block);
            std::vector<double> costs;
            const double seconds = 30.0;
            const size_t total = static_cast<size_t>(seconds * fs);
            const auto start = std::chrono::steady_clock::now();
            for (size_t pos = 0; pos + block <= total; pos += block) {
                const size_t off = pos % (x.size() - block);
                const auto t0 = std::chrono::steady_clock::now();
                e.process(&x[off], out.data(), block);
                costs.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::sort(costs.begin(), costs.end());
            double mean = 0;
            for (double c : costs) mean += c;
            mean /= costs.size();
            const double deadline = 1e6 * block / fs;
            std::printf("| %.0f Hz | %d | %.0f us | %.1f us | %.1f us | %.1f us | %.4f |\n", fs, block, deadline, mean,
                        costs[static_cast<size_t>(0.99 * (costs.size() - 1))], costs.back(), elapsed / seconds);
        }
    }
    std::printf("\nHost measurement only. On-device numbers come from the app's Test Mode.\n");
    return 0;
}
