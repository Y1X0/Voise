"""Roadmap step 5: cross-utterance training. Does it make the model clone people it has NEVER heard better?

Goal of the project: a model that clones any new consenting voice; the 16 pool voices are training material. So the
test is on unseen people only. Two folds; in each, 4 people are left out completely and the model learns from the other
12. Two adapters are trained from the BASE model with the same layers, steps, learning rate and seed:
  same   the current recipe (voise_learn.py): the prompt is the start of the same clip the model must continue
  cross  cross-utterance: the prompt is ANOTHER clip of the same person, and the model must say clip B's words in
         that voice (what zero-shot cloning asks of it)
Then base, same and cross each clone the 4 unseen people: reference = their most typical whole clips up to ~10 s,
3 held-out sentences (as in steps 2/3; no held-out text in any reference), seeds 0 and 1, 32 steps,
class_temperature 0.5. Criteria are fixed in README.md before the run.

Usage: python xutt.py run OUT_DIR       (Kaggle GPU; resumable per fold and arm)
       python xutt.py analyze OUT_DIR RESULTS_DIR
"""
import glob, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "training/scripts")
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
import voice_clone_tts as V  # noqa: E402
import voise_learn as L  # noqa: E402
from ref_duration_full import clips, split, build, WavLMSV, unit, boot_ci, MODEL, ASR, SV, WAVLM_SV, POOL, SR  # noqa: E402

FOLDS, UNSEEN_PER_FOLD, STEPS, LR, SEEDS, REF_S = 2, 4, 600, 1e-5, (0, 1), 10.0
ARMS = ("base", "same", "cross")


def folds(people):
    p = sorted(people)
    random.Random(0).shuffle(p)
    return [p[k * UNSEEN_PER_FOLD:(k + 1) * UNSEEN_PER_FOLD] for k in range(FOLDS)]


class CrossProcessor:
    """OmniVoice's sample processor with the prompt length given by the sample (clip A, whole) instead of a random
    share of the same clip. Masking, condition dropout and the language tag are the library's own."""

    def __init__(self, proc):
        self.p = proc

    def __call__(self, s):
        if "prompt_len" not in s:
            return self.p(s)
        keep = self.p.prompt_ratio_range
        r = (s["prompt_len"] + 0.5) / s["audio_tokens"].shape[1]       # int(n * r) == prompt_len
        self.p.prompt_ratio_range = (r, r)
        try:
            return self.p(s)
        finally:
            self.p.prompt_ratio_range = keep


def train(arm, train_people, model_dir, out_path, device):
    import librosa, soundfile as sf, torch
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.models.omnivoice import OmniVoice
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    fe = AutoFeatureExtractor.from_pretrained(os.path.join(model_dir, "audio_tokenizer"))
    codec = HiggsAudioV2TokenizerModel.from_pretrained(os.path.join(model_dir, "audio_tokenizer")).eval().to(device)
    by = {}
    for p in train_people:
        test, rest, _ = split(clips(p))
        for c in rest:                                        # held-out sentences of every person stay unseen
            y, _ = sf.read(c["wav"], dtype="float32")
            y = librosa.resample(y, orig_sr=SR, target_sr=L.TSR)
            with torch.inference_mode():
                t = codec.encode(fe(raw_audio=y, sampling_rate=L.TSR, return_tensors="pt")["input_values"].to(device)
                                 ).audio_codes.squeeze(0).cpu()
            by.setdefault(p, []).append(dict(tokens=t, text=c["text"]))
    del codec
    model = OmniVoice.from_pretrained(model_dir, device_map=device)
    proc = CrossProcessor(OmniVoiceSampleProcessor(
        text_tokenizer=model.text_tokenizer, num_channels=model.config.num_audio_codebook,
        audio_mask_id=model.config.audio_mask_id, prompt_ratio_range=(0.0, 0.3), mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.1, language_ratio=0.8, use_pinyin_ratio=0.0, instruct_ratio=0.0, only_instruct_ratio=0.0))
    flat = [(p, i) for p in sorted(by) for i in range(len(by[p]))]

    def sample():
        p, i = random.choice(flat)
        b = by[p][i]
        if arm == "same" or len(by[p]) < 2:
            return {"audio_tokens": b["tokens"], "label": {"text": b["text"], "language_id": "apc"}}
        a = by[p][random.choice([k for k in range(len(by[p])) if k != i])]
        # like cloning: the reference clip's words, then the new words; the reference audio is the prompt
        return {"audio_tokens": torch.cat([a["tokens"], b["tokens"]], 1), "prompt_len": a["tokens"].shape[1],
                "label": {"text": a["text"] + " " + b["text"], "language_id": "apc"}}

    layers = model.llm.layers
    first = len(layers) - L.TOP_LAYERS
    names = [n for n, _ in model.named_parameters()
             if n.startswith(("llm.norm.", "audio_heads.")) or any(n.startswith(f"llm.layers.{k}.") for k in range(first, len(layers)))]
    params = dict(model.named_parameters())
    for q in model.parameters():
        q.requires_grad_(False)
    for n in names:
        params[n].requires_grad_(True)
    opt = torch.optim.AdamW([params[n] for n in names], lr=LR, weight_decay=0.01)
    model.train()
    for k in range(1, STEPS + 1):
        b = proc(sample())
        o = model(input_ids=b["input_ids"][None].to(device), audio_mask=b["audio_mask"][None].to(device),
                  labels=b["labels"][None].to(device))
        loss = o.loss if hasattr(o, "loss") else o["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.0)
        opt.step(); opt.zero_grad()
        if k % 100 == 0:
            print("train", arm, k, round(float(loss), 3), flush=True)
    torch.save(dict(weights={n: params[n].detach().cpu().clone() for n in names}, arm=arm, people=sorted(by),
                    steps=STEPS, lr=LR), out_path)
    del model, opt
    torch.cuda.empty_cache() if device == "cuda" else None


def evaluate(arm, weights_path, unseen, everyone, model_dir, device, ecapa, wavlm):
    import librosa, soundfile as sf, torch
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
    model = OmniVoice.from_pretrained(model_dir, device_map=device)
    if weights_path:
        st = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.to(device) for k, v in st["weights"].items()}, strict=False)
    model.eval()
    sr = getattr(model, "sampling_rate", None) or L.TSR
    out = {}
    for p in unseen:
        test, rest, _ = split(clips(p))
        e_rest = {c["wav"]: ecapa(sf.read(c["wav"], dtype="float32")[0]) for c in rest}
        cen = unit([e_rest[c["wav"]].numpy() for c in rest[:30]])
        ranked = sorted(rest, key=lambda c: -float(e_rest[c["wav"]].numpy() @ cen))
        parts = build(ranked, REF_S)
        assert not {V.normalize_ar(c["text"]) for c in parts} & {V.normalize_ar(t["text"]) for t in test}
        gap = np.zeros(int(0.3 * SR), np.float32)
        ref = np.concatenate(sum([[sf.read(c["wav"], dtype="float32")[0], gap] for c in parts], [])[:-1])
        ref_wav = f"/tmp/_xutt_ref_{p}.wav"
        sf.write(ref_wav, ref, SR, subtype="PCM_16")
        prompt = model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=" ".join(c["text"] for c in parts))
        os.remove(ref_wav)
        takes = []
        for t in test:
            for seed in SEEDS:
                torch.manual_seed(seed)
                with torch.no_grad():
                    y = model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                       generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=sr, target_sr=SR)
                takes.append(dict(clip=os.path.basename(t["wav"]), seed=seed,
                                  cer=round(V.cer(t["text"], V.transcribe(y16, ASR)), 4),
                                  ecapa=[round(float(v), 5) for v in ecapa(y16)],
                                  wavlm=[round(float(v), 5) for v in wavlm(y16)]))
        out[p] = dict(reference_seconds=round(len(ref) / SR, 1), takes=takes)
        print("eval", arm, p, round(float(np.mean([k["cer"] for k in takes])), 3), flush=True)
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return out


def run(out_dir):
    import soundfile as sf, torch
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    everyone = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    enc = V.load_speaker_encoder(SV)
    ecapa = lambda y: V.speaker_embedding(enc, y)
    wavlm = WavLMSV(WAVLM_SV)
    if "_people" not in raw:                                  # every person's centres and real held-out clips
        raw["_people"] = {}
        for p in everyone:
            test, rest, _ = split(clips(p))
            ys = [sf.read(c["wav"], dtype="float32")[0] for c in rest[:30]]
            reals = [sf.read(t["wav"], dtype="float32")[0] for t in test]
            raw["_people"][p] = dict(
                centre_ecapa=[round(float(v), 6) for v in unit([ecapa(y).numpy() for y in ys])],
                centre_wavlm=[round(float(v), 6) for v in unit([wavlm(y).numpy() for y in ys])],
                reals=[dict(ecapa=[round(float(v), 5) for v in ecapa(y)], wavlm=[round(float(v), 5) for v in wavlm(y)])
                       for y in reals])
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    for k, unseen in enumerate(folds(everyone)):
        fk = f"fold{k}"
        raw.setdefault(fk, dict(unseen=unseen, arms={}))
        train_people = [p for p in everyone if p not in unseen]
        for arm in ARMS:
            if arm in raw[fk]["arms"]:
                continue
            w = None
            if arm != "base":
                w = os.path.join(out_dir, f"{fk}_{arm}.pt")
                if not os.path.exists(w):
                    train(arm, train_people, MODEL, w, device)
            raw[fk]["arms"][arm] = evaluate(arm, w, unseen, everyone, MODEL, device, ecapa, wavlm)
            json.dump(raw, open(path, "w"), ensure_ascii=False)
    print("DONE run")


def analyze(out_dir, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    raw = json.load(open(os.path.join(out_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    A = lambda v: np.asarray(v, np.float64)
    cen = {p: A(v["centre_ecapa"]) for p, v in P.items()}
    cenw = {p: A(v["centre_wavlm"]) for p, v in P.items()}
    ceil = {p: float(np.mean([A(r["ecapa"]) @ cen[p] for r in v["reals"]])) for p, v in P.items()}
    ho = {p: unit([A(r["ecapa"]) for r in v["reals"]]) for p, v in P.items()}
    how = {p: unit([A(r["wavlm"]) for r in v["reals"]]) for p, v in P.items()}
    per = {arm: {} for arm in ARMS}
    for fk in sorted(k for k in raw if k.startswith("fold")):
        for arm in ARMS:
            for p, d in raw[fk]["arms"][arm].items():
                T = d["takes"]
                m = unit([A(t["ecapa"]) for t in T])
                sims = {o: float(m @ cen[o]) for o in cen}
                other = max((o for o in cen if o != p), key=sims.get)
                mw = unit([A(t["wavlm"]) for t in T])
                simw = {o: float(mw @ cenw[o]) for o in cenw}
                per[arm][p] = dict(closeness=100 * float(np.mean([A(t["ecapa"]) @ cen[p] for t in T])) / ceil[p],
                                   heldout=100 * float(np.mean([A(t["ecapa"]) @ ho[p] for t in T])),
                                   heldout_wavlm=100 * float(np.mean([A(t["wavlm"]) @ how[p] for t in T])),
                                   cer=100 * float(np.mean([t["cer"] for t in T])),
                                   margin=sims[p] - sims[other], identified=max(sims, key=sims.get) == p,
                                   wavlm_identified=max(simw, key=simw.get) == p)
    ps = sorted(per["cross"])
    keys = ("closeness", "heldout", "heldout_wavlm", "cer", "margin")
    mean = {arm: {k: round(float(np.mean([per[arm][p][k] for p in ps])), 3) for k in keys} for arm in ARMS}
    for arm in ARMS:
        mean[arm]["identified"] = sum(per[arm][p]["identified"] for p in ps)
        mean[arm]["wavlm_identified"] = sum(per[arm][p]["wavlm_identified"] for p in ps)
    paired = {f"{x}-{y}": {k: dict(mean=round(float(np.mean([per[x][p][k] - per[y][p][k] for p in ps])), 3),
                                   ci95=boot_ci([per[x][p][k] - per[y][p][k] for p in ps])) for k in keys}
              for x, y in (("cross", "same"), ("cross", "base"), ("same", "base"))}
    d = paired["cross-same"]
    fails = []
    if not d["heldout"]["ci95"][0] > 0:
        fails.append(f"no gain over the current recipe against the unseen people's held-out clips {d['heldout']}")
    if not d["closeness"]["mean"] > 0:
        fails.append(f"closeness not higher {d['closeness']}")
    if not d["heldout_wavlm"]["ci95"][1] > 0:
        fails.append(f"WavLM-SV lower {d['heldout_wavlm']}")
    if d["cer"]["mean"] > 0.5:
        fails.append(f"letter error up {d['cer']['mean']} points")
    if mean["cross"]["identified"] < mean["same"]["identified"]:
        fails.append("fewer unseen people identified")
    if not paired["cross-base"]["heldout"]["mean"] > 0:
        fails.append("not better than the base model")
    res = dict(protocol=__doc__, unseen_people=ps, folds={k: raw[k]["unseen"] for k in raw if k.startswith("fold")},
               per_person={arm: {p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per[arm][p].items()}
                                 for p in ps} for arm in ARMS},
               mean=mean, paired=paired, decision=dict(accepted=not fails, failed=fails))
    json.dump(res, open(os.path.join(res_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(mean=mean, paired=paired, decision=res["decision"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
