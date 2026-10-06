#!/usr/bin/env bash
# Fetch the OFFLINE-EVALUATION model artifacts from their official sources into models/
# (git-ignored). Every archive/checkpoint is checked against the SHA-256 recorded when it
# was first downloaded (2026-10-06); a mismatch aborts and the file is deleted.
# Nothing here is used by the Android app. Licenses: see docs/NEURAL_MODEL_ACQUISITION.md.
#
#   scripts/fetch_models.sh            # everything (~5 GB on disk)
#   scripts/fetch_models.sh knnvc asv  # a subset: knnvc asv resnet b3 whisper
set -euo pipefail
cd "$(dirname "$0")/.."
M=models
DL=$M/_downloads
mkdir -p "$DL"

fetch() {  # url sha256 dest
  local url=$1 sha=$2 dest=$3
  if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status; then
    echo "ok (cached)  $dest"; return
  fi
  mkdir -p "$(dirname "$dest")"
  curl -sSL --fail -m 3600 -o "$dest.part" "$url"
  if ! echo "$sha  $dest.part" | sha256sum -c --status; then
    rm -f "$dest.part"; echo "SHA-256 MISMATCH: $url" >&2; exit 1
  fi
  mv "$dest.part" "$dest"; echo "ok           $dest"
}

clone() {  # url commit dest
  [ -d "$3/.git" ] && { echo "ok (present) $3"; return; }
  GIT_LFS_SKIP_SMUDGE=1 git clone -q "$1" "$3"
  git -C "$3" -c advice.detachedHead=false checkout -q "$2"
  echo "ok           $3 @ $2"
}

want() { [ $# -eq 0 ] || [[ " ${SEL} " == *" $1 "* ]]; }
SEL="${*:-}"
GH=https://github.com

if [ -z "$SEL" ] || want knnvc; then  # kNN-VC (MIT)
  clone $GH/bshall/knn-vc c616845c4e309e24d5927f15adbdf277a3d65358 $M/knn-vc
  fetch $GH/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt \
    f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5 $M/torch_hub/hub/checkpoints/prematch_g_02500000.pt
  fetch $GH/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt \
    6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f $M/torch_hub/hub/checkpoints/WavLM-Large.pt
fi

if [ -z "$SEL" ] || want asv; then  # VoicePrivacy 2024 ASV evaluator (ECAPA, LibriSpeech-360; GPL-3.0)
  if [ ! -f $M/exp/asv_orig/embedding_model.ckpt ]; then
    fetch $GH/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/pre_model.zip/asv_orig.zip \
      f0abf60e569cb60a6f945a1a0a48e1a7d1aabe973f956dc3a872a3e686a7bf5e $DL/asv_orig.zip
    unzip -oq $DL/asv_orig.zip -d $M
  else echo "ok (present) $M/exp/asv_orig"; fi
fi

if [ -z "$SEL" ] || want resnet; then  # SA-toolkit ResNet ASV trained on VoxCeleb1 (GPL-3.0 repo)
  fetch $GH/deep-privacy/SA-toolkit/releases/download/resnet_v1/final.jit \
    07c354bc1f601df7bf7771fca158c33da272461347d88c7c754968ae2e012ab4 $M/satools_resnet_v1/final.jit
  fetch $GH/deep-privacy/SA-toolkit/releases/download/resnet_v1/cfg.configs.resnet \
    a20530cca9341f6f8b952fc106005c5346876a9a310607aac0e2692e88f96d70 $M/satools_resnet_v1/cfg.configs.resnet
fi

if [ -z "$SEL" ] || want b3; then  # VoicePrivacy 2024 B3 / STTTS (GPL-3.0)
  clone $GH/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024 0ef5506911ca6dd142b6f713adf328051879336d $M/vpc2024
  clone $GH/snakers4/silero-vad 6478567951ae5c9979ad7b234185b5515f4be7a1 $M/silero-vad
  if [ ! -f $M/vpc2024/exp/sttts_models/tts/HiFiGAN_combined/best.pt ]; then
    R=$GH/DigitalPhonetics/speaker-anonymization/releases/download/v2.0
    fetch $R/anonymization.zip 0e662a713fb3f01b25000b7d1fa65ff5dd58a39ed7922703df26c8b7a9ba7e32 $DL/anonymization.zip
    fetch $R/asr.zip 14dd433addca14a57a67d1ad838c830ea1a5cb4070dc6a5be95b14ab7a86133d $DL/asr.zip
    fetch $R/tts.zip c9a40069693d98c4d7a9c17646aa670ded668c0f0cc376175fc26d4fb706b8c1 $DL/tts.zip
    for z in asr tts anonymization; do unzip -oq $DL/$z.zip -d $M/vpc2024/exp/sttts_models; done
  else echo "ok (present) $M/vpc2024/exp/sttts_models"; fi
  echo "B3 also needs: apt espeak-ng libportaudio2; a venv with speechbrain==0.5.16 espnet==202310 (see docs)"
fi

if [ -z "$SEL" ] || want whisper; then  # Whisper small.en (MIT) as ONNX by k2-fsa/sherpa-onnx (Apache-2.0)
  if [ ! -f $M/sherpa-onnx-whisper-small.en/small.en-encoder.onnx ]; then
    fetch $GH/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2 \
      0cdba2b8aaab69e04847f3427cc9709574112e67913a1a84b7fec3a8729faa9a $DL/whisper-small.en.tar.bz2
    tar xjf $DL/whisper-small.en.tar.bz2 -C $M
  else echo "ok (present) $M/sherpa-onnx-whisper-small.en"; fi
fi
