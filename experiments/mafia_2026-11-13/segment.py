"""Step 1 for a multi-speaker recording: short single-turn speech segments, each with a speaker embedding (ECAPA)
and what was said (dialect Whisper), so speakers can be grouped and named from how they address each other.
Usage: python segment.py WAV OUT_DIR"""
import json, os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A, voice_clone_tts as V, voise_learn as L  # noqa: E402
SR = 16000
wav, out = sys.argv[1:3]
y = sf.read(wav, dtype="float32")[0]
vad, enc = A.load_vad(), V.load_speaker_encoder("models/ecapa-voxceleb")
regs = []
for a, b in A.speech_regions(y, vad, min_silence_s=0.2):
    regs += L.split_long(y, a, b, max_s=8.0, target_s=6.0) if b - a > 8 * SR else [(a, b)]
regs = [(a, b) for a, b in regs if b - a >= 1.0 * SR]
print(len(regs), "segments", round(sum(b - a for a, b in regs) / SR / 60, 1), "min", flush=True)
segs, embs = [], []
for k, (a, b) in enumerate(regs):
    seg = y[a:b]
    embs.append(np.asarray(V.speaker_embedding(enc, seg)))
    segs.append(dict(i=k, a=round(a / SR, 2), b=round(b / SR, 2), text=V.transcribe(seg, "models/whisper-turbo-dialectal-v2-ct2")))
    if k % 100 == 0:
        print(k, segs[-1], flush=True)
np.save(os.path.join(out, "emb.npy"), np.stack(embs))
json.dump(segs, open(os.path.join(out, "segments.json"), "w"), ensure_ascii=False, indent=0)
print("DONE")
