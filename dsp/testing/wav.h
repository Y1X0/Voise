// Tiny WAV reader/writer for the host tools (16-bit PCM and 32-bit float, mono
// mix-down on read). Not used by the Android app.
#pragma once

#include <string>
#include <vector>

namespace voiceanon {
namespace testing {

bool readWav(const std::string& path, std::vector<float>& samples, int& sampleRate);
bool writeWav(const std::string& path, const std::vector<float>& samples, int sampleRate);

}  // namespace testing
}  // namespace voiceanon
