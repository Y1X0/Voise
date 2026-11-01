"""Voise learning loop: the model keeps learning from the consented voices it is given.

  add     split a consenting person's recordings into clean 2-14 s clips and add them to the learning pool
  train   learn from the whole pool; the new version is kept ONLY if it beats the current one on held-out clips
  revoke  remove one person from the pool and drop the learned version (it contained their voice)
  status  what is in the pool and which version is in use

Design (each point fixes something we measured):
  * Learning builds on itself: each `train` continues from the current learned version and is judged against it,
    so the model keeps improving as the pool grows. Only `revoke` (or --from-scratch) restarts from the original
    Lahgtna, because a revoked voice must not stay inside the weights.
  * Only the top transformer layers, the final norm and the audio heads learn; the rest stays frozen, so the model
    keeps what it knows (a full-model run on 30 h of Syrian speech drifted the accent to Syrian).
  * Clips are pooled across people, so the model learns the dialect and the way people talk, not one person's
    microphone (training on one person's 2.5 min of voice notes made that person's clone worse).
  * Quality gate on the way in (speech length, background noise, Whisper must understand it) and an
    acceptance gate on the way out (held-out masked-token loss must drop, and Whisper must not hear the test
    sentences worse than before). A version that fails the gate is discarded and the previous one stays.
  * A person is only added if their consent record is complete, not revoked, and allows training
    ("training": true, or "train" in its scope).

The learned version is a small file (only the trained weights) that voice_clone_tts.py loads with --adapter.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_clone_tts as V  # noqa: E402

SR, TSR = 16000, 24000
TOP_LAYERS = 6
TEST_TEXTS = ["أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.",
              "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟",
              "شو رأيك نطلع بكرا عالبلد؟ بدي أشتري كم غرض وبعدين منشرب قهوة."]


def allows_training(rec):
    return rec.get("training") is True or "train" in str(rec.get("scope", "")).lower()


def segments(y, vad, min_s=2.0, max_s=14.0):
    """Cut a recording at its pauses into speech spans of min_s..max_s seconds."""
    import audio_preprocess as A
    regs = A.speech_regions(y, vad)
    out, cur = [], None
    for a, b in regs:
        if cur is None:
            cur = [a, b]
        elif b - cur[0] <= max_s * SR:
            cur[1] = b
        else:
            out.append(cur); cur = [a, b]
    if cur:
        out.append(cur)
    return [(max(0, a - 800), min(len(y), b + 800)) for a, b in out if (b - a) >= min_s * SR and (b - a) <= max_s * SR]


def noise_db(y, vad):
    """Speech level minus the level in the pauses (a rough signal-to-noise ratio, dB); None if no pauses."""
    import audio_preprocess as A
    regs = A.speech_regions(y, vad)
    gaps = [y[b:a] for (_, b), (a, _) in zip(regs[:-1], regs[1:]) if a - b > 1600]
    if not gaps:
        return None
    sp = np.concatenate([y[a:b] for a, b in regs])
    db = lambda x: 10 * np.log10(float(np.mean(x ** 2)) + 1e-12)
    return db(sp) - db(np.concatenate(gaps))


def cmd_add(a):
    import soundfile as sf
    import audio_preprocess as A
    from unit_vocoder_pilot import load_audio_16k
    rec, sha = V.load_consent(a.consent)
    if not allows_training(rec):
        raise SystemExit("refusing: this consent record does not allow training (add \"training\": true only if the "
                         "person agreed to their voice being used to train the model)")
    person = V.safe_slug(a.person or rec["voice_owner"])
    dst = os.path.join(a.pool, person)
    V.check_out_dir(dst)
    vad = A.load_vad()
    files = sorted(glob.glob(os.path.join(a.src, "*.wav")) + glob.glob(os.path.join(a.src, "*.flac")))
    added = rejected = 0
    for p in files:
        y, _ = load_audio_16k(p, min_sr=8000)
        snr = noise_db(y, vad)
        if snr is not None and snr < a.min_snr:
            print(f"  rejected {os.path.basename(p)}: too much background noise ({snr:.0f} dB < {a.min_snr})"); rejected += 1
            continue
        for s, e in segments(y, vad):
            clip = y[s:e] / max(1e-6, float(np.abs(y[s:e]).max())) * 0.9
            h = hashlib.sha256(clip.astype(np.float32).tobytes()).hexdigest()[:16]
            if os.path.exists(os.path.join(dst, h + ".wav")):
                continue
            text = V.transcribe(clip, a.asr_dir)
            if len(V.normalize_ar(text)) < 4:
                rejected += 1
                continue
            sf.write(os.path.join(dst, h + ".wav"), clip, SR, subtype="PCM_16")
            json.dump(dict(text=text, source=os.path.basename(p), consent_record=a.consent, consent_sha256=sha,
                           added_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
                      open(os.path.join(dst, h + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
            added += 1
    print(f"{person}: +{added} clips, {rejected} rejected; pool now {len(glob.glob(os.path.join(dst, '*.wav')))} clips")


def pool_items(pool):
    items = []
    for wav in sorted(glob.glob(os.path.join(pool, "*", "*.wav"))):
        meta = json.load(open(wav[:-4] + ".json", encoding="utf-8"))
        rec, _ = V.load_consent(meta["consent_record"])   # a consent revoked since is refused here
        items.append(dict(person=os.path.basename(os.path.dirname(wav)), wav=wav, text=meta["text"]))
    return items


def cmd_train(a):
    import librosa
    import soundfile as sf
    import torch
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.models.omnivoice import OmniVoice
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    items = pool_items(a.pool)
    people = sorted({i["person"] for i in items})
    rel = lambda w: os.path.relpath(w, a.pool)
    seen = set()
    if os.path.exists(a.adapter) and not a.from_scratch:
        seen = set(torch.load(a.adapter, map_location="cpu", weights_only=True).get("clips", []))
    new_min = sum(sf.info(i["wav"]).duration for i in items if rel(i["wav"]) not in seen) / 60
    if new_min < a.min_new_minutes:
        print(f"only {new_min:.1f} min of new voice since the last learning (need {a.min_new_minutes}); nothing to do")
        return
    minutes = sum(sf.info(i["wav"]).duration for i in items) / 60
    print(f"pool: {len(items)} clips, {minutes:.1f} min, {len(people)} people")
    if len(items) < 20:
        raise SystemExit("not enough to learn from yet: need at least 20 clean clips")
    # hold out ~15% of every person's clips for the acceptance gate
    held, train = [], []
    for p in people:
        mine = [i for i in items if i["person"] == p]
        random.Random(0).shuffle(mine)
        k = max(1, round(len(mine) * 0.15)) if len(mine) >= 4 else 0
        held += mine[:k]; train += mine[k:]

    fe = AutoFeatureExtractor.from_pretrained(os.path.join(a.model_dir, "audio_tokenizer"))
    codec = HiggsAudioV2TokenizerModel.from_pretrained(os.path.join(a.model_dir, "audio_tokenizer")).eval()

    def tokens(i):
        y, _ = sf.read(i["wav"], dtype="float32")
        y = librosa.resample(y, orig_sr=SR, target_sr=TSR)
        with torch.inference_mode():
            c = codec.encode(fe(raw_audio=y, sampling_rate=TSR, return_tensors="pt")["input_values"]).audio_codes.squeeze(0)
        return {"audio_tokens": c, "label": {"text": i["text"], "language_id": "apc"}}

    train_s, held_s = [tokens(i) for i in train], [tokens(i) for i in held]
    del codec
    model = OmniVoice.from_pretrained(a.model_dir, device_map="cpu")
    prev_people, generation = [], 0
    if os.path.exists(a.adapter) and not a.from_scratch:
        st = torch.load(a.adapter, map_location="cpu", weights_only=True)
        model.load_state_dict(st["weights"], strict=False)
        prev_people, generation = st.get("people", []), int(st.get("generation", 0))
        print(f"continuing from the current learned version (generation {generation})")
    proc = OmniVoiceSampleProcessor(text_tokenizer=model.text_tokenizer, num_channels=model.config.num_audio_codebook,
                                    audio_mask_id=model.config.audio_mask_id, prompt_ratio_range=(0.0, 0.3),
                                    mask_ratio_range=(0.0, 1.0), drop_cond_ratio=0.1, language_ratio=0.8,
                                    use_pinyin_ratio=0.0, instruct_ratio=0.0, only_instruct_ratio=0.0)

    def loss_of(s):
        b = proc(s)
        o = model(input_ids=b["input_ids"][None], audio_mask=b["audio_mask"][None], labels=b["labels"][None])
        return o.loss if hasattr(o, "loss") else o["loss"]

    def held_loss():
        model.eval()
        st = random.getstate(); tst = torch.random.get_rng_state()
        random.seed(1); torch.manual_seed(1)
        with torch.no_grad():
            v = float(np.mean([float(loss_of(s)) for s in held_s for _ in range(3)]))
        random.setstate(st); torch.random.set_rng_state(tst)
        return round(v, 4)

    ref = held[0]["wav"] if held else train[0]["wav"]
    ref_text = next(i["text"] for i in items if i["wav"] == ref)

    def heard_cer():
        model.eval()
        errs = []
        for t in TEST_TEXTS:
            torch.manual_seed(0)
            with torch.no_grad():
                y = model.generate(text=t, language="apc", ref_audio=ref, ref_text=ref_text, duration=V.target_duration(t))
            y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
            errs.append(V.cer(t, V.transcribe(librosa.resample(y, orig_sr=TSR, target_sr=SR), a.asr_dir)))
        return round(float(np.mean(errs)), 4)

    report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  people=people, train_clips=len(train), held_clips=len(held), minutes=round(minutes, 1),
                  continued_from_generation=generation if prev_people or generation else None)
    report["held_loss_before"] = held_loss()
    report["cer_before"] = heard_cer()
    print("before:", report["held_loss_before"], report["cer_before"], flush=True)

    layers = model.llm.layers
    first = len(layers) - TOP_LAYERS
    names = [n for n, _ in model.named_parameters()
             if n.startswith(("llm.norm.", "audio_heads.")) or any(n.startswith(f"llm.layers.{k}.") for k in range(first, len(layers)))]
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for n in names:
        params[n].requires_grad_(True)
    opt = torch.optim.AdamW([params[n] for n in names], lr=a.lr, weight_decay=0.01)
    steps = a.steps or min(600, 15 * len(train_s))
    model.train()
    t0 = time.time()
    for k in range(1, steps + 1):
        loss = loss_of(random.choice(train_s))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.0)
        opt.step(); opt.zero_grad()
        if k % 50 == 0:
            print(f"step {k}/{steps} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)
    report["steps"] = steps
    report["held_loss_after"] = held_loss()
    report["cer_after"] = heard_cer()
    print("after:", report["held_loss_after"], report["cer_after"], flush=True)

    ok = report["held_loss_after"] < report["held_loss_before"] and report["cer_after"] <= report["cer_before"] + 0.02
    report["accepted"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.adapter)), exist_ok=True)
    if ok:
        tmp = a.adapter + ".tmp"
        torch.save({"base": os.path.abspath(a.model_dir), "people": sorted(set(people) | set(prev_people)),
                    "generation": generation + 1, "clips": sorted(rel(i["wav"]) for i in items), "weights": {n: params[n].detach().clone() for n in names}}, tmp)
        os.replace(tmp, a.adapter)
        print("accepted: the model learned from the pool and the new version is in use")
    else:
        print("not accepted: the new version was not better on held-out clips; the previous version stays in use")
    json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(a.adapter)), "last_train.json"), "w"), indent=1)


def cmd_revoke(a):
    person = V.safe_slug(a.person)
    d = os.path.join(a.pool, person)
    if os.path.isdir(d):
        shutil.rmtree(d)
    if os.path.exists(a.adapter):
        people = __import__("torch").load(a.adapter, map_location="cpu", weights_only=True).get("people", [])
        if person in people:
            os.remove(a.adapter)
            print("the learned version contained this voice and was deleted; run train again to rebuild it without them")
    print(f"removed {person} from the learning pool")


def cmd_status(a):
    for d in sorted(glob.glob(os.path.join(a.pool, "*"))):
        print(f"{os.path.basename(d)}: {len(glob.glob(os.path.join(d, '*.wav')))} clips")
    rep = os.path.join(os.path.dirname(os.path.abspath(a.adapter)), "last_train.json")
    print("learned version:", "in use" if os.path.exists(a.adapter) else "none (original model)")
    if os.path.exists(rep):
        print("last training:", open(rep).read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["add", "train", "revoke", "status"])
    ap.add_argument("--pool", default=os.path.join(V.RUNS, "learn", "pool"))
    ap.add_argument("--adapter", default=os.path.join(V.RUNS, "learn", "adapter.pt"))
    ap.add_argument("--model-dir", default="models/lahgtna-omnivoice-v2")
    ap.add_argument("--asr-dir", default="models/whisper-large-v3-turbo")
    ap.add_argument("--consent")
    ap.add_argument("--src", help="folder of the person's recordings (add)")
    ap.add_argument("--person")
    ap.add_argument("--min-snr", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-new-minutes", type=float, default=0.0,
                    help="learn only if at least this much new voice arrived since the current version")
    ap.add_argument("--from-scratch", action="store_true", help="ignore the current learned version and start from Lahgtna")
    a = ap.parse_args(argv)
    {"add": cmd_add, "train": cmd_train, "revoke": cmd_revoke, "status": cmd_status}[a.command](a)


if __name__ == "__main__":
    main()
