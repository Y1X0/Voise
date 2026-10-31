"""Clone each friend from ONE reference clip taken from each condition; score against held-out RAW clips.

The reference clip and the held-out split are the same ones used by the raw baseline
(runs/friends/baseline/results.json: ref; held-out = a third of the other clips, shuffled with seed 0),
so the only thing that changes between conditions is how the reference was cleaned.

Usage: python clone_eval.py RESTORED_DIR BASELINE_DIR ECAPA_DIR OUT_DIR
"""
import glob
import json
import os
import random
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A  # noqa: E402

SR = 16000
TEXT = "أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص."
SPEAKERS = ("khaled", "muhannad", "mohammad", "majd", "shahd")


def main(root, base_dir, ecapa_dir, out):
    from omnivoice.models.omnivoice import OmniVoice
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    os.makedirs(out, exist_ok=True)
    base = json.load(open(os.path.join(base_dir, "results.json")))
    ec = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir + "_sv", run_opts={"device": "cpu"})
    pr = WhisperProcessor.from_pretrained("models/whisper-large-v3-turbo")
    asr = WhisperForConditionalGeneration.from_pretrained("models/whisper-large-v3-turbo", torch_dtype=torch.float32).eval()
    tts = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
    tsr = getattr(tts, "sampling_rate", None) or 24000
    vad = A.load_vad()

    def load(p):
        y, sr = sf.read(p, dtype="float32")
        return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y

    def speech(y):
        return A.compact(y, A.speech_regions(y, vad), max_pause_s=0.3)[0]

    def emb(y):
        with torch.no_grad():
            return torch.nn.functional.normalize(ec.encode_batch(torch.as_tensor(y)[None]).squeeze(), dim=-1)

    def tr(y):
        ids = asr.generate(pr(y, sampling_rate=SR, return_tensors="pt").input_features, language="ar",
                           task="transcribe", num_beams=5)
        return pr.batch_decode(ids, skip_special_tokens=True)[0].strip()

    res = {}
    for spk in SPEAKERS:
        raw = {os.path.basename(p): speech(load(p)) for p in sorted(glob.glob(os.path.join(root, "raw", spk, "*.wav")))}
        raw = {k: v for k, v in raw.items() if len(v) >= 2 * SR}
        names = [n for n in raw if n != base[spk]["ref"]]
        random.Random(0).shuffle(names)
        test = names[: max(1, len(names) // 3)]
        te = [emb(raw[n]) for n in test]
        score = lambda y: round(float(np.mean([float(emb(y) @ t) for t in te])), 3)
        res[spk] = {"ref": base[spk]["ref"], "test": len(test),
                    "raw": score(load(os.path.join(base_dir, f"SYNTHETIC_{spk}.wav")))}
        for cond in ("sepformer", "sidon"):
            ref = speech(load(os.path.join(root, cond, spk, base[spk]["ref"])))[: 12 * SR]
            ref = ref / max(1e-6, float(np.abs(ref).max())) * 0.9
            rw = os.path.join(out, f"{spk}_{cond}_ref.wav")
            sf.write(rw, ref, SR, subtype="PCM_16")
            torch.manual_seed(0)
            y = tts.generate(text=TEXT, language="apc", ref_audio=rw, ref_text=tr(ref))
            y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
            y = y / max(1e-6, float(np.abs(y).max())) * 0.9
            sf.write(os.path.join(out, f"SYNTHETIC_{spk}_{cond}.wav"), y, tsr, subtype="PCM_16")
            g = librosa.resample(y, orig_sr=tsr, target_sr=SR)
            res[spk][cond] = score(g)
            res[spk][cond + "_heard"] = tr(g)
        print(spk, res[spk], flush=True)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:5])
