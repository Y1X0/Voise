"""Per-speaker adaptation of Lahgtna-OmniVoice on a few minutes of one consenting speaker's clips (CPU).

Only the top N transformer layers, the final norm and the audio heads are trained; everything else stays frozen,
so the model keeps its dialect knowledge while learning the speaker. Training samples follow OmniVoice's own
recipe (OmniVoiceSampleProcessor: random prompt prefix, random mask ratio, loss on masked audio tokens only).
The held-out clips (a third of the non-reference clips, shuffled with seed 0, as in the earlier experiments) are
never trained on, and the adapted model is scored on them exactly like the zero-shot baseline.

Usage: python adapt.py SPEAKER FRIENDS_DIR ECAPA_DIR OUT_DIR [steps] [top_layers] [lr]
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

SR, TSR = 16000, 24000
MODEL = "models/lahgtna-omnivoice-v2"
TEXTS = ["أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.",
         "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟"]


def main(spk, friends, ecapa_dir, out, steps=300, top_layers=6, lr=2e-5):
    steps, top_layers, lr = int(steps), int(top_layers), float(lr)
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.models.omnivoice import OmniVoice
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel, WhisperForConditionalGeneration, WhisperProcessor
    torch.manual_seed(0); random.seed(0)
    os.makedirs(out, exist_ok=True)
    base = json.load(open(os.path.join(friends, "baseline", "results.json")))
    vad = A.load_vad()
    pr = WhisperProcessor.from_pretrained("models/whisper-large-v3-turbo")
    asr = WhisperForConditionalGeneration.from_pretrained("models/whisper-large-v3-turbo", torch_dtype=torch.float32).eval()
    ec = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir + "_sv", run_opts={"device": "cpu"})

    def load(p, sr=SR):
        y, s = sf.read(p, dtype="float32")
        return librosa.resample(y, orig_sr=s, target_sr=sr) if s != sr else y

    def speech(y):
        return A.compact(y, A.speech_regions(y, vad), max_pause_s=0.3)[0]

    def tr(y):
        ids = asr.generate(pr(y, sampling_rate=SR, return_tensors="pt").input_features, language="ar",
                           task="transcribe", num_beams=5)
        return pr.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def emb(y):
        with torch.no_grad():
            return torch.nn.functional.normalize(ec.encode_batch(torch.as_tensor(y)[None]).squeeze(), dim=-1)

    # ---- split exactly as the baseline did
    clips = {os.path.basename(p): speech(load(p)) for p in sorted(glob.glob(os.path.join(friends, "by_speaker", spk, "*.wav")))}
    clips = {k: v for k, v in clips.items() if len(v) >= 2 * SR}
    names = [n for n in clips if n != base[spk]["ref"]]
    random.Random(0).shuffle(names)
    test = names[: max(1, len(names) // 3)]
    train = [n for n in clips if n not in test]
    te = [emb(clips[n]) for n in test]
    print(f"{spk}: train {len(train)} clips ({sum(len(clips[n]) for n in train) / SR / 60:.1f} min), test {len(test)}", flush=True)

    # ---- audio tokens + transcripts for the training clips
    fe = AutoFeatureExtractor.from_pretrained(os.path.join(MODEL, "audio_tokenizer"))
    codec = HiggsAudioV2TokenizerModel.from_pretrained(os.path.join(MODEL, "audio_tokenizer")).eval()
    samples = []
    for n in train:
        y16 = clips[n]
        y24 = librosa.resample(y16, orig_sr=SR, target_sr=TSR)
        with torch.inference_mode():
            codes = codec.encode(fe(raw_audio=y24, sampling_rate=TSR, return_tensors="pt")["input_values"]).audio_codes.squeeze(0)
        samples.append({"audio_tokens": codes.cpu(), "label": {"text": tr(y16), "language_id": "apc"}})
    del codec

    model = OmniVoice.from_pretrained(MODEL, device_map="cpu")
    proc = OmniVoiceSampleProcessor(text_tokenizer=model.text_tokenizer, num_channels=model.config.num_audio_codebook,
                                    audio_mask_id=model.config.audio_mask_id, prompt_ratio_range=(0.0, 0.3),
                                    mask_ratio_range=(0.0, 1.0), drop_cond_ratio=0.1, language_ratio=0.8,
                                    use_pinyin_ratio=0.0, instruct_ratio=0.0, only_instruct_ratio=0.0)

    def clone(tag):
        model.eval()
        ref = os.path.join(friends, "baseline", f"{spk}_ref.wav")
        rt = tr(load(ref))
        r = {}
        for i, t in enumerate(TEXTS):
            torch.manual_seed(0)
            with torch.no_grad():
                y = model.generate(text=t, language="apc", ref_audio=ref, ref_text=rt)
            y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
            y = y / max(1e-6, float(np.abs(y).max())) * 0.9
            sf.write(os.path.join(out, f"SYNTHETIC_{spk}_{tag}_{i}.wav"), y, TSR, subtype="PCM_16")
            g = librosa.resample(y, orig_sr=TSR, target_sr=SR)
            r[f"sim_{i}"] = round(float(np.mean([float(emb(g) @ t_) for t_ in te])), 3)
            r[f"heard_{i}"] = tr(g)
        r["sim"] = round((r["sim_0"] + r["sim_1"]) / 2, 3)
        return r

    res = {"speaker": spk, "train_clips": len(train), "test_clips": len(test), "steps": steps,
           "top_layers": top_layers, "lr": lr, "before": clone("before")}
    print("before", res["before"], flush=True)

    # ---- train only the top layers, final norm and audio heads
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.llm.layers
    first = len(layers) - top_layers
    names = [n for n, _ in model.named_parameters()
             if n.startswith(("llm.norm.", "audio_heads.")) or any(n.startswith(f"llm.layers.{i}.") for i in range(first, len(layers)))]
    params = dict(model.named_parameters())
    trainable = [params[n] for n in names]
    for p in trainable:
        p.requires_grad_(True)
    print("trainable params (M):", round(sum(p.numel() for p in trainable) / 1e6, 1), flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train()
    t0, losses = time.time(), []
    for step in range(1, steps + 1):
        s = proc(random.choice(samples))
        o = model(input_ids=s["input_ids"][None], audio_mask=s["audio_mask"][None], labels=s["labels"][None])
        loss = o.loss if hasattr(o, "loss") else o["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad()
        losses.append(float(loss))
        if step % 25 == 0:
            print(f"step {step} loss {np.mean(losses[-25:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    res["train_secs"] = round(time.time() - t0)
    res["loss_first_last"] = [round(float(np.mean(losses[:25])), 3), round(float(np.mean(losses[-25:])), 3)]
    res["after"] = clone("after")
    print("after", res["after"], flush=True)
    torch.save({n: params[n].detach().clone() for n in names}, os.path.join(out, f"{spk}_adapter.pt"))
    json.dump(res, open(os.path.join(out, f"{spk}_results.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
