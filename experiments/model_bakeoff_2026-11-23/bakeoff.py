"""Base-model bake-off (2026-11-23): is there a stronger zero-shot voice cloner than the one Voise uses? No training.

Models (each with its own documented inference settings, never tuned on these results):
  lahgtna   oddadmix/lahgtna-omnivoice-v2 (what Voise uses), WITHOUT our adapter: the adapter was trained on these
            16 people, so it is excluded to keep the test speaker-disjoint from our training
  omnivoice k2-fsa/OmniVoice (the base Lahgtna was fine-tuned from)
  voxcpm2   openbmb/VoxCPM2 ("ultimate cloning": prompt audio + its transcript + the same clip as reference),
            cfg 2.0, 10 timesteps
  audar     audarai/Audar-TTS-V1-Flash (bf16 transformers path, base NeuCodec), temperature 0.7 (the card's setting
            for neutral delivery), top_k 40, top_p 0.9, repetition_penalty 1.1
OmniVoice-family settings: guidance 2.0, class_temperature 0.5, 32 steps (the tool's).

The 16 consented people are speaker-disjoint from OUR training, not guaranteed unseen by the pretrained models.

Protocols:
  P1 (fair, same input for all): the person's most ECAPA-typical whole clips up to ~10 s (inside every model's
     recommended range, e.g. Audar 5-15 s) and their transcript.
  P2 (best input per model): OmniVoice-family with ~19 s (what Voise found best); VoxCPM2 and Audar keep P1 (their
     cards recommend short clips).
3 held-out sentences per person (same split as steps 2-5; no held-out text in any reference), seeds 0 and 1.
Seed-0 P1 takes are kept as SYNTHETIC_*.wav with the watermark, for listening; everything else is numbers.

Usage: python bakeoff.py run OUT_DIR MODEL[,MODEL] ; python bakeoff.py analyze OUT_DIR RESULTS_DIR
"""
import gc, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "training/scripts")
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
import voice_clone_tts as V  # noqa: E402
from ref_duration_full import clips, split, build, WavLMSV, unit, boot_ci, eer, ASR, SV, WAVLM_SV, POOL, SR  # noqa: E402

MODELS = ("lahgtna", "omnivoice", "voxcpm2", "audar")
SEEDS = (0, 1)
P1_S, P2_S = 10.0, 19.0
OMNI = {"lahgtna": "models/lahgtna-omnivoice-v2", "omnivoice": "models/omnivoice-base"}


# ---------- model backends: generate(ref_wav, ref_text, text, seed) -> (y, sr) ----------
class Omni:
    def __init__(self, path):
        import torch
        from omnivoice.models.omnivoice import OmniVoice
        self.m = OmniVoice.from_pretrained(path, device_map="cuda" if torch.cuda.is_available() else "cpu")
        self.sr = getattr(self.m, "sampling_rate", None) or 24000
        self.cache = {}

    def __call__(self, ref_wav, ref_text, text, seed):
        import torch
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        if ref_wav not in self.cache:
            self.cache[ref_wav] = self.m.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)
        torch.manual_seed(seed)
        with torch.no_grad():
            y = self.m.generate(text=text, language="apc", voice_clone_prompt=self.cache[ref_wav],
                                generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
        return np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze(), self.sr


class VoxCPM2:
    def __init__(self):
        from voxcpm import VoxCPM
        self.m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        self.sr = self.m.tts_model.sample_rate

    def __call__(self, ref_wav, ref_text, text, seed):
        import torch
        torch.manual_seed(seed)
        y = self.m.generate(text=text, prompt_wav_path=ref_wav, prompt_text=ref_text, reference_wav_path=ref_wav,
                            cfg_value=2.0, inference_timesteps=10)
        return np.asarray(y, np.float32).squeeze(), self.sr


class Audar:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from neucodec import NeuCodec
        repo = "audarai/Audar-TTS-V1-Flash"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(repo, subfolder="transformers")
        self.m = AutoModelForCausalLM.from_pretrained(repo, subfolder="transformers",
                                                      torch_dtype=torch.bfloat16).eval().to(dev)
        self.codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to(dev)
        self.sr, self.cache = 24000, {}

    def __call__(self, ref_wav, ref_text, text, seed):
        import librosa, torch
        if ref_wav not in self.cache:
            w, _ = librosa.load(ref_wav, sr=16000, mono=True)
            with torch.no_grad():
                codes = self.codec.encode_code(torch.from_numpy(w)[None, None, :]).squeeze().tolist()
            self.cache[ref_wav] = "".join(f"<|speech_{c}|>" for c in codes)
        prompt = ("user: Convert the text to speech:"
                  f"<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>"
                  f"<|REF_SPEECH_START|>{self.cache[ref_wav]}<|REF_SPEECH_END|>"
                  f"<|TARGET_TEXT_START|>{text}<|TARGET_TEXT_END|>"
                  "\nassistant:<|TARGET_CODES_START|>")
        ids = self.tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.m.device)
        tce = self.tok.convert_tokens_to_ids("<|TARGET_CODES_END|>")
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.m.generate(ids, max_new_tokens=1500, do_sample=True, temperature=0.7, top_k=40, top_p=0.9,
                                  repetition_penalty=1.1, min_new_tokens=50, eos_token_id=tce, pad_token_id=151643)
            txt = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            codes = [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", txt)]
            y = self.codec.decode_code(torch.tensor(codes)[None, None, :].to(self.m.device)).cpu().numpy()[0, 0, :]
        return np.asarray(y, np.float32), self.sr


def load(name):
    if name in OMNI:
        return Omni(OMNI[name])
    return {"voxcpm2": VoxCPM2, "audar": Audar}[name]()


# ---------- run ----------
def run(out_dir, names):
    import librosa, soundfile as sf, torch
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    everyone = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    enc = V.load_speaker_encoder(SV)
    ecapa = lambda y: V.speaker_embedding(enc, y)
    wavlm = WavLMSV(WAVLM_SV)
    if "_people" not in raw:                        # centres, held-out reals, and the two references per person
        raw["_people"] = {}
        ref_dir = os.path.join(out_dir, "refs")
        os.makedirs(ref_dir, exist_ok=True)
        for p in everyone:
            test, rest, _ = split(clips(p))
            texts = {V.normalize_ar(t["text"]) for t in test}
            ys = {c["wav"]: sf.read(c["wav"], dtype="float32")[0] for c in rest}
            cen = unit([ecapa(ys[c["wav"]]).numpy() for c in rest[:30]])
            ranked = sorted(rest, key=lambda c: -float(ecapa(ys[c["wav"]]).numpy() @ cen))
            refs = {}
            for tag, s in (("p1", P1_S), ("p2", P2_S)):
                parts = build(ranked, s)
                assert not {V.normalize_ar(c["text"]) for c in parts} & texts
                gap = np.zeros(int(0.3 * SR), np.float32)
                ref = np.concatenate(sum([[ys[c["wav"]], gap] for c in parts], [])[:-1])
                w = os.path.join(ref_dir, f"{p}_{tag}.wav")
                sf.write(w, ref, SR, subtype="PCM_16")
                refs[tag] = dict(wav=w, text=" ".join(c["text"] for c in parts), seconds=round(len(ref) / SR, 1))
            reals = [sf.read(t["wav"], dtype="float32")[0] for t in test]
            raw["_people"][p] = dict(
                refs=refs, test=[dict(clip=os.path.basename(t["wav"]), text=t["text"]) for t in test],
                centre_ecapa=[round(float(v), 6) for v in cen],
                centre_wavlm=[round(float(v), 6) for v in unit([wavlm(ys[c["wav"]]).numpy() for c in rest[:30]])],
                reals=[dict(ecapa=[round(float(v), 5) for v in ecapa(y)], wavlm=[round(float(v), 5) for v in wavlm(y)])
                       for y in reals])
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    for name in names:
        protos = ("p1", "p2") if name in OMNI else ("p1",)
        todo = [(p, pr) for p in everyone for pr in protos if f"{p}/{pr}" not in raw.get(name, {})]
        if not todo:
            continue
        gen = load(name)
        raw.setdefault(name, {})
        for p, pr in todo:
            P = raw["_people"][p]
            ref = P["refs"][pr]
            takes = []
            for i, t in enumerate(P["test"]):
                for seed in SEEDS:
                    try:
                        y, sr = gen(ref["wav"], ref["text"], t["text"], seed)
                    except Exception as e:                       # record, never silently drop
                        takes.append(dict(clip=t["clip"], seed=seed, error=repr(e)[:300]))
                        continue
                    y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=sr, target_sr=SR)
                    takes.append(dict(clip=t["clip"], seed=seed, seconds=round(len(y16) / SR, 2),
                                      cer=round(V.cer(t["text"], V.transcribe(y16, ASR)), 4),
                                      ecapa=[round(float(v), 5) for v in ecapa(y16)],
                                      wavlm=[round(float(v), 5) for v in wavlm(y16)]))
                    if seed == 0 and pr == "p1":                 # for listening: watermarked, SYNTHETIC_ named
                        wm, _ = V.watermark(y, sr)
                        sf.write(os.path.join(audio_dir, f"SYNTHETIC_{name}_{p}_{i}.wav"), wm, sr, subtype="PCM_16")
            raw[name][f"{p}/{pr}"] = dict(reference_seconds=ref["seconds"], takes=takes)
            json.dump(raw, open(path, "w"), ensure_ascii=False)
            ok = [k for k in takes if "error" not in k]
            print("R", name, p, pr, len(ok), "ok", round(float(np.mean([k["cer"] for k in ok])), 3) if ok else None, flush=True)
        del gen
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("DONE run", names)


# ---------- analyze ----------
def analyze(out_dirs, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    raw = {}
    for d in out_dirs.split(","):
        r = json.load(open(os.path.join(d, "raw.json"), encoding="utf-8"))
        for k, v in r.items():
            if k == "_people":
                raw.setdefault("_people", v)
            else:
                raw.setdefault(k, {}).update(v)
    P = raw["_people"]
    ps = sorted(P)
    A = lambda v: np.asarray(v, np.float64)
    cen = {p: A(P[p]["centre_ecapa"]) for p in ps}
    cenw = {p: A(P[p]["centre_wavlm"]) for p in ps}
    ceil = {p: float(np.mean([A(r["ecapa"]) @ cen[p] for r in P[p]["reals"]])) for p in ps}
    ho = {p: unit([A(r["ecapa"]) for r in P[p]["reals"]]) for p in ps}
    how = {p: unit([A(r["wavlm"]) for r in P[p]["reals"]]) for p in ps}

    def measure(name, proto):
        per, means, errors = {}, {}, 0
        for p in ps:
            T = [t for t in raw[name][f"{p}/{proto}"]["takes"] if "error" not in t]
            errors += len(raw[name][f"{p}/{proto}"]["takes"]) - len(T)
            if not T:
                return None
            means[p] = unit([A(t["ecapa"]) for t in T])
            mw = unit([A(t["wavlm"]) for t in T])
            sims = {o: float(means[p] @ cen[o]) for o in ps}
            simw = {o: float(mw @ cenw[o]) for o in ps}
            other = max((o for o in ps if o != p), key=sims.get)
            per[p] = dict(closeness=100 * float(np.mean([A(t["ecapa"]) @ cen[p] for t in T])) / ceil[p],
                          heldout=100 * float(np.mean([A(t["ecapa"]) @ ho[p] for t in T])),
                          heldout_wavlm=100 * float(np.mean([A(t["wavlm"]) @ how[p] for t in T])),
                          cer=100 * float(np.mean([t["cer"] for t in T])),
                          margin=sims[p] - sims[other], identified=max(sims, key=sims.get) == p,
                          wavlm_identified=max(simw, key=simw.get) == p)
        tgt, non = [], []
        for p in ps:
            for t in raw[name][f"{p}/{proto}"]["takes"]:
                if "error" in t:
                    continue
                for o in ps:
                    (tgt if o == p else non).append(float(A(t["ecapa"]) @ cen[o]))
        return dict(per=per, errors=errors, eer=eer(np.asarray(tgt), np.asarray(non)),
                    spread=float(np.mean([means[a] @ means[b] for i, a in enumerate(ps) for b in ps[i + 1:]])),
                    mean={k: round(float(np.mean([per[p][k] for p in ps])), 3)
                          for k in ("closeness", "heldout", "heldout_wavlm", "cer", "margin")},
                    identified=sum(per[p]["identified"] for p in ps),
                    wavlm_identified=sum(per[p]["wavlm_identified"] for p in ps))

    arms = {}
    for name in MODELS:
        if name not in raw:
            continue
        for proto in (("p1", "p2") if name in OMNI else ("p1",)):
            if all(f"{p}/{proto}" in raw[name] for p in ps):
                m = measure(name, proto)
                if m:
                    arms[f"{name}/{proto}"] = m
    best = lambda name: arms.get(f"{name}/p2") or arms.get(f"{name}/p1")
    current = best("lahgtna")

    def vs(a, b):
        d = lambda k: [b["per"][p][k] - a["per"][p][k] for p in ps]
        return {k: dict(mean=round(float(np.mean(d(k))), 3), ci95=boot_ci(d(k)))
                for k in ("closeness", "heldout", "heldout_wavlm", "cer", "margin")}

    decisions = {}
    for name in MODELS:                              # each model's best protocol vs Voise's best (lahgtna P2)
        if name == "lahgtna" or not best(name) or not current:
            continue
        pr = vs(current, best(name))
        fails = []
        if not pr["heldout"]["ci95"][0] > 0:
            fails.append(f"held-out ECAPA gain not established {pr['heldout']}")
        if not pr["heldout_wavlm"]["ci95"][0] > 0:
            fails.append(f"held-out WavLM-SV gain not established {pr['heldout_wavlm']}")
        if pr["cer"]["mean"] > 1.0:
            fails.append(f"letter error up {pr['cer']['mean']} points")
        if best(name)["identified"] < current["identified"]:
            fails.append(f"fewer identified ({best(name)['identified']} vs {current['identified']})")
        decisions[name] = dict(paired_vs_current=pr, stronger=not fails, failed=fails)
    fair = {name: vs(arms["lahgtna/p1"], arms[f"{name}/p1"]) for name in MODELS
            if name != "lahgtna" and f"{name}/p1" in arms and "lahgtna/p1" in arms}
    res = dict(protocol=__doc__, people=ps,
               arms={k: dict(mean=v["mean"], eer=v["eer"], spread=round(v["spread"], 4), identified=v["identified"],
                             wavlm_identified=v["wavlm_identified"], errors=v["errors"],
                             per_person={p: {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v["per"][p].items()}
                                         for p in ps}) for k, v in arms.items()},
               fair_p1_vs_lahgtna=fair, best_vs_current=decisions)
    json.dump(res, open(os.path.join(res_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(means={k: dict(v["mean"], identified=v["identified"], eer=v["eer"]) for k, v in arms.items()},
                          decisions=decisions), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3].split(","))
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
