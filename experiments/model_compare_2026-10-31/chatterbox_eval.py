"""One-clip cloning with Chatterbox Multilingual V3 (ResembleAI/chatterbox, MIT; outputs carry Perth watermarks),
scored exactly like the Lahgtna-OmniVoice baseline: same reference clip per speaker, same held-out raw clips
(a third of the other clips, shuffled with seed 0), ECAPA cosine, Whisper large-v3-turbo transcript.

Usage: python chatterbox_eval.py CHATTERBOX_DIR FRIENDS_DIR ECAPA_DIR OUT_DIR
"""
import glob
import json
import os
import random
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A  # noqa: E402

SR = 16000
TEXT = "أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص."
SPEAKERS = ("khaled", "muhannad", "mohammad", "majd", "shahd")


def load_chatterbox(d):
    from pathlib import Path
    from safetensors.torch import load_file
    from chatterbox.mtl_tts import S3Gen, T3, ChatterboxMultilingualTTS, MTLTokenizer, T3Config, VoiceEncoder
    d = Path(d)
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(d / "ve.safetensors"))
    t3 = T3(T3Config.multilingual())
    st = load_file(d / "t3_mtl23ls_v3.safetensors")
    t3.load_state_dict(st["model"][0] if "model" in st else st)
    s3 = S3Gen()
    # the two missing keys are the tokenizer's mel filters / window buffers, which S3Gen builds itself
    missing = s3.load_state_dict(load_file(d / "s3gen_v3.safetensors"), strict=False).missing_keys
    assert set(missing) <= {"tokenizer._mel_filters", "tokenizer.window"}, missing
    tok = MTLTokenizer(str(d / "grapheme_mtl_merged_expanded_v1.json"))
    return ChatterboxMultilingualTTS(t3.eval(), s3.eval(), ve.eval(), tok, "cpu", conds=None)


def main(cb_dir, friends, ecapa_dir, out):
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    os.makedirs(out, exist_ok=True)
    base = json.load(open(os.path.join(friends, "baseline", "results.json")))
    ec = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir + "_sv", run_opts={"device": "cpu"})
    pr = WhisperProcessor.from_pretrained("models/whisper-large-v3-turbo")
    asr = WhisperForConditionalGeneration.from_pretrained("models/whisper-large-v3-turbo", torch_dtype=torch.float32).eval()
    cb = load_chatterbox(cb_dir)
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
        raw = {os.path.basename(p): speech(load(p)) for p in sorted(glob.glob(os.path.join(friends, "by_speaker", spk, "*.wav")))}
        raw = {k: v for k, v in raw.items() if len(v) >= 2 * SR}
        names = [n for n in raw if n != base[spk]["ref"]]
        random.Random(0).shuffle(names)
        te = [emb(raw[n]) for n in names[: max(1, len(names) // 3)]]
        score = lambda y: round(float(np.mean([float(emb(y) @ t) for t in te])), 3)
        ref = os.path.join(friends, "baseline", f"{spk}_ref.wav")
        res[spk] = {"lahgtna": score(load(os.path.join(friends, "baseline", f"SYNTHETIC_{spk}.wav")))}
        for tag, kw in (("chatterbox", {}), ("chatterbox_expr", {"exaggeration": 0.7, "cfg_weight": 0.3})):
            torch.manual_seed(0)
            t0 = time.time()
            y = cb.generate(TEXT, language_id="ar", audio_prompt_path=ref, **kw).squeeze().numpy()
            y = y / max(1e-6, float(np.abs(y).max())) * 0.9
            sf.write(os.path.join(out, f"SYNTHETIC_{spk}_{tag}.wav"), y, cb.sr, subtype="PCM_16")
            g = librosa.resample(y, orig_sr=cb.sr, target_sr=SR)
            res[spk][tag] = score(g)
            res[spk][tag + "_heard"] = tr(g)
            res[spk][tag + "_secs"] = round(time.time() - t0)
        print(spk, res[spk], flush=True)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:5])
