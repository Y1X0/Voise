#!/bin/bash
# Voise on a laptop (WSL/Ubuntu): one entry point for the consent-gated TTS tool.
#
#   ./voise.sh setup                  install packages and download the models (once; resumes if cut)
#   ./voise.sh prepare                read C:\Voise_Voices\<Person>\ (consent.* + voice notes) into runs/
#   ./voise.sh speak <person> <text>  speak <text> in <person>'s voice -> runs/voices/<person>/output/
#   ./voise.sh fast <person> <text>   the same, about 2x faster per take (16 steps instead of 32)
#   ./voise.sh serve                  keep the models loaded (leave it open in its own window) and open the demo
#                                     page http://127.0.0.1:8765 in the browser; speak/fast then skip the
#                                     ~5 min of loading and reuse each person's prepared voice
#   ./voise.sh list                   people who are ready
#   ./voise.sh revoke <person>        consent withdrawn: delete that person's recordings, outputs and learned voice
#   ./voise.sh learn                  learn now from every voice in the learning pool
#
# Learning: if a person's folder also holds a file named train_ok (any extension, e.g. a screenshot of them
# agreeing to their voice being used to TRAIN the model), prepare adds their clean clips to the learning pool and,
# once 3 or more minutes of new voice have arrived, the model learns from everything in the pool. Each learned
# version builds on the last one and replaces it only if it tests better. speak always uses the latest version.
#
# A folder without a consent.* file (screenshot or text) is skipped; the tool itself refuses to run without
# a complete, unrevoked consent record. Every output is named SYNTHETIC_*.
set -e
cd "$(dirname "$0")"
SRC=${SRC:-/mnt/c/Voise_Voices}
SV=""; [ -d models/ecapa-voxceleb ] && SV="--sv-dir models/ecapa-voxceleb"
MOS=models/utmos/utmos22_strong_step7459_v1.pt; [ -f "$MOS" ] && SV="$SV --mos $MOS"
ASR=models/whisper-turbo-dialectal-v2-ct2
[ -d "$ASR" ] || ASR=models/whisper-large-v3-turbo
[ -d "$ASR" ] || ASR=models/whisper-small

fetch() {  # repo revision dest patterns...
  python3 - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst, *pat = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=dst, allow_patterns=pat)
EOF
}

case "$1" in
setup)
  sudo apt install -y ffmpeg
  pip install -q --no-deps omnivoice==0.2.1
  pip install -q pydub tensorboardx webdataset accelerate silero-vad soundfile librosa audioop-lts transformers \
    faster-whisper resemble-perth speechbrain "setuptools<81"   # Perth's watermarker needs pkg_resources
  export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
  until fetch oddadmix/lahgtna-omnivoice-v2 55c38a613e31 models/lahgtna-omnivoice-v2 '*.json' '*.safetensors' '*.jinja' \
     && fetch eustlb/higgs-audio-v2-tokenizer 528e871c2a26 models/lahgtna-omnivoice-v2/audio_tokenizer '*.json' '*.safetensors' \
     && fetch openai/whisper-large-v3-turbo 41f01f3fe87f models/whisper-large-v3-turbo '*.json' '*.safetensors' '*.txt'
  do echo "download interrupted, retrying in 15 s..."; sleep 15; done
  # speaker encoder (Apache-2.0): picks the reference that sounds most like the person and ranks takes by voice
  until fetch speechbrain/spkrec-ecapa-voxceleb 0f99f2d0ebe8 models/ecapa-voxceleb '*'
  do echo "download interrupted, retrying in 15 s..."; sleep 15; done
  # naturalness predictor (SpeechMOS UTMOS22-strong, MIT), reported for every take
  [ -f "$MOS" ] || { mkdir -p models/utmos && curl -sSL -o "$MOS" \
    https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt; }
  echo "38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff  $MOS" | sha256sum -c --quiet || { rm -f "$MOS"; echo "UTMOS download corrupt; rerun setup"; }
  # Arabic-dialect Whisper (Apache-2.0), converted to a fast int8 CPU model
  until fetch oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2 b12b09bc6601 models/whisper-turbo-dialectal-v2 '*.json' '*.safetensors'
  do echo "download interrupted, retrying in 15 s..."; sleep 15; done
  [ -f models/whisper-turbo-dialectal-v2-ct2/model.bin ] || {
    cp models/whisper-large-v3-turbo/preprocessor_config.json models/whisper-turbo-dialectal-v2/
    ct2-transformers-converter --model models/whisper-turbo-dialectal-v2 --output_dir models/whisper-turbo-dialectal-v2-ct2 \
      --quantization int8 --copy_files tokenizer.json preprocessor_config.json; }
  echo "setup done"
  ;;
prepare)
  for dir in "$SRC"/*/; do
    [ -d "$dir" ] || continue
    person=$(basename "$dir" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '_')
    img=$(ls "$dir"consent.* 2>/dev/null | head -1)
    if [ -z "$img" ]; then echo "SKIPPED $person: no consent.* file in $dir"; continue; fi
    mkdir -p runs/consent "runs/voices/$person"
    cp "$img" "runs/consent/$person.${img##*.}"
    if [ ! -f "runs/consent/$person.json" ]; then
      cat > "runs/consent/$person.json" <<JSON
{
 "voice_owner": "$(basename "$dir")",
 "date": "$(date +%F)",
 "evidence": "runs/consent/$person.${img##*.}",
 "scope": "testing and development of Voise (graduation project) only",
 "prohibited": ["impersonation", "attributing words they did not say", "publishing or sharing their voice"],
 "revocable": "any time; on revocation run ./voise.sh revoke $person"
}
JSON
    fi
    if ls "$dir"train_ok* >/dev/null 2>&1; then
      python3 - "runs/consent/$person.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
if not d.get("training"):
    d["training"] = True
    d["scope"] = d["scope"].replace(" only", "") + ", including training the Voise model"
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
      learn_people="$learn_people $person"
    fi
    n=0
    for f in "$dir"*; do
      [ -f "$f" ] || continue
      case "$(basename "${f,,}")" in consent.*|train_ok*|._*|*.jpg|*.jpeg|*.png|*.txt) continue;; esac
      out="runs/voices/$person/$(basename "${f%.*}" | tr -c 'A-Za-z0-9_\n-' '_').wav"
      ffmpeg -nostdin -loglevel error -y -i "$f" -ac 1 -ar 16000 "$out" && n=$((n+1))
    done
    echo "READY $person: $n recording(s)"
  done
  for person in $learn_people; do
    python3 training/scripts/voise_learn.py add --consent "runs/consent/$person.json" --src "runs/voices/$person" \
      --person "$person" --asr-dir "$ASR"
  done
  if [ -n "$learn_people" ]; then
    echo "learning from the pool if 3+ minutes of new voice arrived (this can take a while on a laptop)..."
    python3 training/scripts/voise_learn.py train --min-new-minutes 3 --asr-dir "$ASR"
  fi
  ;;
learn)
  python3 training/scripts/voise_learn.py train --asr-dir "$ASR"
  ;;
speak|fast)
  mode=$1; person=$2; shift 2
  [ -n "$person" ] && [ -n "$*" ] || { echo "usage: ./voise.sh speak|fast <person> <text>"; exit 1; }
  if curl -s -m 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
    # the engine is running (./voise.sh serve): models are already loaded, only the sentence is made
    python3 - "$person" "$mode" "$*" <<'PY'
import json, sys, urllib.request
person, mode, text = sys.argv[1:4]
req = urllib.request.Request("http://127.0.0.1:8765/speak", method="POST", headers={"Content-Type": "application/json"},
                             data=json.dumps({"person": person, "text": text, "fast": mode == "fast"}).encode())
try:
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    print(r["wav"], f"({r['seconds']} s, {r['takes']} take(s), heard: {r['heard']})")
except urllib.error.HTTPError as e:
    sys.exit(json.load(e).get("error"))
PY
  else
    extra=""; [ "$mode" = fast ] && extra="--fast"
    python3 training/scripts/voice_clone_tts.py --consent "runs/consent/$person.json" \
      --ref-dir "runs/voices/$person" --model-dir models/lahgtna-omnivoice-v2 --asr-dir "$ASR" \
      --out "runs/voices/$person/output" --adapter runs/learn/adapter.pt $SV $extra --text "$*"
  fi
  command -v explorer.exe >/dev/null && explorer.exe "$(wslpath -w "runs/voices/$person/output")" || true
  ;;
serve)
  python3 training/scripts/voise_server.py --asr-dir "$ASR" $SV &
  pid=$!
  until curl -s -m 2 http://127.0.0.1:8765/health >/dev/null 2>&1; do sleep 3; kill -0 $pid 2>/dev/null || exit 1; done
  command -v explorer.exe >/dev/null && explorer.exe "http://127.0.0.1:8765" || true   # open the page in Windows
  wait $pid
  ;;
list)
  for c in runs/consent/*.json; do [ -f "$c" ] && basename "$c" .json; done
  ;;
revoke)
  person=$2
  [ -n "$person" ] || { echo "usage: ./voise.sh revoke <person>"; exit 1; }
  rm -rf "runs/voices/$person" runs/consent/"$person".*
  python3 training/scripts/voise_learn.py revoke --person "$person"
  echo "deleted $person's recordings, consent record and outputs; also delete $SRC/<their folder>"
  ;;
*)
  sed -n 2,22p "$0"
  ;;
esac
