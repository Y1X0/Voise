#!/bin/bash
# Voise on a laptop (WSL/Ubuntu): one entry point for the consent-gated TTS tool.
#
#   ./voise.sh setup                  install packages and download the models (once; resumes if cut)
#   ./voise.sh prepare                read C:\Voise_Voices\<Person>\ (consent.* + voice notes) into runs/
#   ./voise.sh speak <person> <text>  speak <text> in <person>'s voice -> runs/voices/<person>/output/
#   ./voise.sh list                   people who are ready
#   ./voise.sh revoke <person>        consent withdrawn: delete that person's recordings and outputs
#
# A folder without a consent.* file (screenshot or text) is skipped; the tool itself refuses to run without
# a complete, unrevoked consent record. Every output is named SYNTHETIC_*.
set -e
cd "$(dirname "$0")"
SRC=${SRC:-/mnt/c/Voise_Voices}
ASR=models/whisper-large-v3-turbo
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
  pip install -q pydub tensorboardx webdataset accelerate silero-vad soundfile librosa audioop-lts transformers
  export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
  until fetch oddadmix/lahgtna-omnivoice-v2 55c38a613e31 models/lahgtna-omnivoice-v2 '*.json' '*.safetensors' '*.jinja' \
     && fetch eustlb/higgs-audio-v2-tokenizer 528e871c2a26 models/lahgtna-omnivoice-v2/audio_tokenizer '*.json' '*.safetensors' \
     && fetch openai/whisper-large-v3-turbo 41f01f3fe87f models/whisper-large-v3-turbo '*.json' '*.safetensors' '*.txt'
  do echo "download interrupted, retrying in 15 s..."; sleep 15; done
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
    n=0
    for f in "$dir"*; do
      [ -f "$f" ] || continue
      case "$(basename "${f,,}")" in consent.*|._*|*.jpg|*.jpeg|*.png|*.txt) continue;; esac
      out="runs/voices/$person/$(basename "${f%.*}" | tr -c 'A-Za-z0-9_\n-' '_').wav"
      ffmpeg -nostdin -loglevel error -y -i "$f" -ac 1 -ar 16000 "$out" && n=$((n+1))
    done
    echo "READY $person: $n recording(s)"
  done
  ;;
speak)
  person=$2; shift 2
  [ -n "$person" ] && [ -n "$*" ] || { echo "usage: ./voise.sh speak <person> <text>"; exit 1; }
  python3 training/scripts/voice_clone_tts.py --consent "runs/consent/$person.json" \
    --ref-dir "runs/voices/$person" --model-dir models/lahgtna-omnivoice-v2 --asr-dir "$ASR" \
    --out "runs/voices/$person/output" --text "$*"
  command -v explorer.exe >/dev/null && explorer.exe "$(wslpath -w "runs/voices/$person/output")" || true
  ;;
list)
  for c in runs/consent/*.json; do [ -f "$c" ] && basename "$c" .json; done
  ;;
revoke)
  person=$2
  [ -n "$person" ] || { echo "usage: ./voise.sh revoke <person>"; exit 1; }
  rm -rf "runs/voices/$person" runs/consent/"$person".*
  echo "deleted $person's recordings, consent record and outputs; also delete $SRC/<their folder>"
  ;;
*)
  sed -n 2,10p "$0"
  ;;
esac
