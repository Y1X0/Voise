"""Short and noisy references, clean vs noisy vs cleaned (criteria in README.md, fixed before generation).

Usage:
  python hard_refs.py generate RAW_DIR     # resumable per voice; RAW_DIR under runs/ (git-ignored)
  python hard_refs.py analyze RAW_DIR OUT_DIR
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
import ref_duration_full as R  # noqa: E402

V = R.V
PEOPLE = ("mafia_p106", "mafia2_p152", "mafia_p43")
LENGTHS = (5.0, 19.0)
VERSIONS = ("clean", "noisy", "cleaned")
SNR_DB = 10.0
CENTRES = "runs/ref_duration_full/raw.json"
CLEANER = "speechbrain/sepformer-dns4-16k-enhancement"
GAIN, MIN_VOICES, CER_UP = 2.0, 2, 0.01


def pink(n, rng):
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(w)); f[0] = 1
    y = np.fft.irfft(w / np.sqrt(f), n)
    return (y / np.std(y)).astype(np.float32)


def add_noise(y, snr_db, seed):
    nz = pink(len(y), np.random.default_rng(seed))
    p = float(np.mean(y ** 2))
    return (y + nz * np.sqrt(p / 10 ** (snr_db / 10))).astype(np.float32)


def generate(raw_dir):
    import librosa, soundfile as sf, torch
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    from speechbrain.inference.separation import SepformerSeparation
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    P_all = json.load(open(CENTRES, encoding="utf-8"))["_people"]
    raw["_people"] = {p: P_all[p] for p in PEOPLE}
    eng = V.Engine(R.MODEL, R.ASR, adapter=R.ADAPTER, sv_dir=R.SV)
    ecapa = lambda y: V.speaker_embedding(eng.enc, y)
    wavlm = R.WavLMSV(R.WAVLM_SV)
    cleaner = SepformerSeparation.from_hparams(source=CLEANER, savedir="models/sepformer-dns4-16k")
    for k, person in enumerate(PEOPLE):
        if person in raw:
            continue
        version = eng.use_version_for(os.path.join(R.POOL, person))
        cs = R.clips(person)
        R.check_consent(cs)
        test, rest, _ = R.split(cs)
        P = raw["_people"][person]
        assert [os.path.basename(t["wav"]) for t in test] == [r["clip"] for r in P["reals"]], "split changed"
        cen = torch.tensor(P["centre_ecapa"], dtype=torch.float32)
        e_rest = {c["wav"]: ecapa(sf.read(c["wav"], dtype="float32")[0]) for c in rest}
        ranked = sorted(rest, key=lambda c: -float(e_rest[c["wav"]] @ cen))
        out = dict(version=version, refs={}, takes=[])
        for L in LENGTHS:
            parts = R.build(ranked, L)
            assert not {V.normalize_ar(c["text"]) for c in parts} & {V.normalize_ar(t["text"]) for t in test}
            gap = np.zeros(int(0.3 * R.SR), np.float32)
            clean = np.concatenate(sum([[sf.read(c["wav"], dtype="float32")[0], gap] for c in parts], [])[:-1])
            noisy = add_noise(clean, SNR_DB, seed=1000 * k + int(L))
            with torch.no_grad():
                cleaned = cleaner.separate_batch(torch.from_numpy(noisy)[None]).squeeze().numpy().astype(np.float32)
            cleaned = cleaned[:len(noisy)]
            text = " ".join(c["text"] for c in parts)
            for name, audio in zip(VERSIONS, (clean, noisy, cleaned)):
                audio = audio / max(1e-6, float(np.abs(audio).max())) * 0.9
                out["refs"][f"{int(L)}_{name}"] = dict(seconds=round(len(audio) / R.SR, 2),
                                                       sim_to_clean=round(float(ecapa(audio) @ ecapa(clean)), 4))
                ref_wav = os.path.join(raw_dir, f"_ref_{person}.wav")
                sf.write(ref_wav, audio, R.SR, subtype="PCM_16")
                prompt = eng.model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=text)
                os.remove(ref_wav)
                for i, t in enumerate(test):
                    torch.manual_seed(0)
                    with torch.no_grad():
                        y = eng.model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                               generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                    y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                    y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr,
                                           target_sr=R.SR)
                    heard = V.transcribe(y16, R.ASR)
                    out["takes"].append(dict(length=int(L), version=name, sentence=i,
                                             cer=round(V.cer(t["text"], heard), 4),
                                             ecapa=[round(float(v), 5) for v in ecapa(y16)],
                                             wavlm=[round(float(v), 5) for v in wavlm(y16)]))
                    print("T", person, int(L), name, i, out["takes"][-1]["cer"], flush=True)
        raw[person] = out
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    print("DONE generate")


def analyze(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = json.load(open(os.path.join(raw_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = [p for p in PEOPLE if p in raw]
    A = lambda v: np.asarray(v, np.float64)
    per = {}
    for p in ps:
        ho, how = R.unit([r["ecapa"] for r in P[p]["reals"]]), R.unit([r["wavlm"] for r in P[p]["reals"]])
        for L in LENGTHS:
            for v in VERSIONS:
                T = [t for t in raw[p]["takes"] if t["length"] == int(L) and t["version"] == v]
                per[(p, int(L), v)] = dict(
                    ecapa=100 * float(np.mean([A(t["ecapa"]) @ ho for t in T])),
                    wavlm=100 * float(np.mean([A(t["wavlm"]) @ how for t in T])),
                    cer=float(np.mean([t["cer"] for t in T])),
                    ref_sim_to_clean=raw[p]["refs"][f"{int(L)}_{v}"]["sim_to_clean"])
    mean = lambda L, v, k: float(np.mean([per[(p, L, v)][k] for p in ps]))
    fails, verdict = [], {}
    for L in (5, 19):
        d = [per[(p, L, "cleaned")]["ecapa"] - per[(p, L, "noisy")]["ecapa"] for p in ps]
        dc = mean(L, "cleaned", "cer") - mean(L, "noisy", "cer")
        verdict[L] = dict(gain_mean=round(float(np.mean(d)), 2), voices_up=int(sum(x > 0 for x in d)),
                          cer_change=round(dc, 4))
        if np.mean(d) < GAIN:
            fails.append(f"{L} s: cleaning gain {np.mean(d):+.2f} < +{GAIN}")
        if sum(x > 0 for x in d) < MIN_VOICES:
            fails.append(f"{L} s: cleaning helped only {sum(x > 0 for x in d)} of {len(ps)} voices")
        if dc > CER_UP:
            fails.append(f"{L} s: letter error up {100 * dc:.2f} points")
    res = dict(people=ps, per={f"{p}|{L}|{v}": x for (p, L, v), x in per.items()}, verdict=verdict,
               decision=dict(confirm_on_16=not fails, failed=fails))
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    L_ = ["# Short and noisy references: results (3-voice screen)", "",
          f"Voices: {', '.join(ps)}; 3 held-out sentences, seed 0. Noise: pink, {SNR_DB:.0f} dB SNR. "
          "Criteria: `../README.md`.", "",
          "| reference | held-out ECAPA | held-out WavLM | letter error | reference vs clean (ECAPA) |",
          "|---|---|---|---|---|"]
    for L in (19, 5):
        for v in VERSIONS:
            L_.append(f"| {L} s {v} | {mean(L, v, 'ecapa'):.1f} | {mean(L, v, 'wavlm'):.2f} | "
                      f"{100 * mean(L, v, 'cer'):.1f}% | {mean(L, v, 'ref_sim_to_clean'):.3f} |")
    L_ += ["", "Per voice, held-out ECAPA (clean / noisy / cleaned):", ""]
    for p in ps:
        L_.append(f"- {p}: " + "; ".join(
            f"{L} s {per[(p, L, 'clean')]['ecapa']:.1f} / {per[(p, L, 'noisy')]['ecapa']:.1f} / "
            f"{per[(p, L, 'cleaned')]['ecapa']:.1f}" for L in (19, 5)))
    L_ += ["", f"Noise cost (noisy - clean): 19 s {mean(19, 'noisy', 'ecapa') - mean(19, 'clean', 'ecapa'):+.1f}, "
           f"5 s {mean(5, 'noisy', 'ecapa') - mean(5, 'clean', 'ecapa'):+.1f} points. "
           f"Length cost (5 s - 19 s, clean): {mean(5, 'clean', 'ecapa') - mean(19, 'clean', 'ecapa'):+.1f} points.",
           "", f"**Decision: {'cleaning passes the screen; confirm on all 16 voices' if not fails else 'cleaning does not pass the screen'}**",
           ""] + [f"- {f}" for f in fails]
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
