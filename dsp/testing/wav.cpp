#include "wav.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

namespace voiceanon {
namespace testing {
namespace {

uint32_t rd32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint16_t rd16(const unsigned char* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
void wr32(std::ofstream& f, uint32_t v) {
    const unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                                static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
    f.write(reinterpret_cast<const char*>(b), 4);
}
void wr16(std::ofstream& f, uint16_t v) {
    const unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
    f.write(reinterpret_cast<const char*>(b), 2);
}

}  // namespace

bool readWav(const std::string& path, std::vector<float>& samples, int& sampleRate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) != 0 || std::memcmp(d.data() + 8, "WAVE", 4) != 0)
        return false;
    size_t p = 12;
    int format = 0, channels = 0, bits = 0;
    sampleRate = 0;
    while (p + 8 <= d.size()) {
        const uint32_t size = rd32(&d[p + 4]);
        const size_t body = p + 8;
        if (std::memcmp(&d[p], "fmt ", 4) == 0 && body + 16 <= d.size()) {
            format = rd16(&d[body]);
            channels = rd16(&d[body + 2]);
            sampleRate = static_cast<int>(rd32(&d[body + 4]));
            bits = rd16(&d[body + 14]);
            if (format == 0xFFFE && size >= 26) format = rd16(&d[body + 24]);  // extensible
        } else if (std::memcmp(&d[p], "data", 4) == 0) {
            if (channels <= 0) return false;
            const size_t bytes = std::min<size_t>(size, d.size() - body);
            const int bps = bits / 8;
            const size_t frames = bytes / (bps * channels);
            samples.assign(frames, 0.0f);
            for (size_t i = 0; i < frames; ++i) {
                float acc = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    const unsigned char* s = &d[body + (i * channels + c) * bps];
                    if (format == 1 && bits == 16) acc += static_cast<int16_t>(rd16(s)) / 32768.0f;
                    else if (format == 1 && bits == 24)
                        acc += static_cast<float>((static_cast<int32_t>((s[0] << 8) | (s[1] << 16) | (s[2] << 24)) >> 8) / 8388608.0);
                    else if (format == 3 && bits == 32) {
                        float v;
                        std::memcpy(&v, s, 4);
                        acc += v;
                    } else return false;
                }
                samples[i] = acc / channels;
            }
            return sampleRate > 0;
        }
        p = body + size + (size & 1);
    }
    return false;
}

bool writeWav(const std::string& path, const std::vector<float>& samples, int sampleRate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * 2);
    f.write("RIFF", 4);
    wr32(f, 36 + dataBytes);
    f.write("WAVEfmt ", 8);
    wr32(f, 16);
    wr16(f, 1);
    wr16(f, 1);
    wr32(f, static_cast<uint32_t>(sampleRate));
    wr32(f, static_cast<uint32_t>(sampleRate * 2));
    wr16(f, 2);
    wr16(f, 16);
    f.write("data", 4);
    wr32(f, dataBytes);
    for (float s : samples) {
        const float c = std::min(std::max(s, -1.0f), 1.0f);
        wr16(f, static_cast<uint16_t>(static_cast<int16_t>(std::lround(c * 32767.0f))));
    }
    return static_cast<bool>(f);
}

}  // namespace testing
}  // namespace voiceanon
