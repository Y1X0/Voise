#!/usr/bin/env bash
# Downloads a few openly licensed real speech recordings (not committed to the
# repo) and evaluates every preset with the same streaming engine the app uses.
#
#   scripts/eval_real_speech.sh [build-dir]      -> prints a Markdown table
#
# Sources (all 16 kHz English):
#   CMU ARCTIC: aew (US male), axb (Indian-English female), a0007 (pysptk sample)
#   MS-SNSD clean test speech (MIT), SpeechBrain test sample (Apache-2.0)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/dsp/build}"
AUDIO="$ROOT/eval-audio"
OUT="$ROOT/eval-out"
mkdir -p "$AUDIO" "$OUT"

fetch() { [ -s "$AUDIO/$2" ] || curl -fsSL -o "$AUDIO/$2" "$1"; }
fetch https://raw.githubusercontent.com/LCAV/pyroomacoustics/master/examples/input_samples/cmu_arctic_us_aew_a0001.wav arctic_aew_male.wav
fetch https://raw.githubusercontent.com/LCAV/pyroomacoustics/master/examples/input_samples/cmu_arctic_us_axb_a0004.wav arctic_axb_female.wav
fetch https://raw.githubusercontent.com/r9y9/pysptk/master/pysptk/example_audio_data/arctic_a0007.wav arctic_a0007_male.wav
fetch https://raw.githubusercontent.com/microsoft/MS-SNSD/master/clean_test/clnsp1.wav mssnsd_clnsp1_male.wav
fetch https://raw.githubusercontent.com/speechbrain/speechbrain/develop/tests/samples/single-mic/example1.wav speechbrain_example1.wav

echo "| recording | preset | dir | pitch (st) | formant | env. corr | inton. corr | MFCC cos | LTAS dB | clicks | dropouts | clipped |"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
for f in "$AUDIO"/*.wav; do
  for preset in natural balanced strong; do
    for dir in auto up; do
      line="$("$BUILD/voiceanon_eval" --preset "$preset" --direction "$dir" --out "$OUT" "$f")"
      python3 - "$line" "$preset" "$dir" <<'PY'
import json, sys
d = json.loads(sys.argv[1]); m = d["metrics"]
fr = "n/a" if m["formantRatio"] < 0 else "x%.3f" % m["formantRatio"]
print("| %s | %s | %s | %+.2f | %s | %.3f | %.3f | %.3f | %.2f | %d | %d | %d |" % (
    d["name"].rsplit("_", 1)[0], sys.argv[2], sys.argv[3], m["pitchShiftSemitones"], fr,
    m["envelopeCorrelation"], m["intonationCorrelation"], m["mfccCosine"], m["ltasDistanceDb"],
    m["newClicks"], m["dropouts"], m["clippedSamples"]))
PY
    done
  done
done
