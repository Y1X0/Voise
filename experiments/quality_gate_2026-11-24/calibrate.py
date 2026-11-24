"""Calibrate the reference noise warning (rule in README.md, fixed before generation).

Usage:
  python calibrate.py generate RAW_DIR      # resumable per voice; RAW_DIR under runs/ (git-ignored)
  python calibrate.py analyze RAW_DIR OUT_DIR
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
sys.path.insert(0, os.path.join(HERE, "..", "hard_refs_2026-11-24"))
import ref_duration_full as R  # noqa: E402
from hard_refs import add_noise, PEOPLE, CENTRES  # noqa: E402

V = R.V
LEVELS = ("clean", 20, 15, 10)
WARN_LOSS, BAD_LOSS = 5.0, 10.0


def generate(raw_dir):
    import librosa, soundfile as sf, torch
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    P_all = json.load(open(CENTRES, encoding="utf-8"))["_people"]
    raw["_people"] = {p: P_all[p] for p in PEOPLE}
    eng = V.Engine(R.MODEL, R.ASR, adapter=R.ADAPTER, sv_dir=R.SV)
    ecapa = lambda y: V.speaker_embedding(eng.enc, y)
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
        parts = R.build(sorted(rest, key=lambda c: -float(e_rest[c["wav"]] @ cen)), 19.0)
        assert not {V.normalize_ar(c["text"]) for c in parts} & {V.normalize_ar(t["text"]) for t in test}
        text = " ".join(c["text"] for c in parts)
        gap = np.zeros(int(0.3 * R.SR), np.float32)
        out = dict(version=version, refs={}, takes=[])
        for lv in LEVELS:
            ys = [sf.read(c["wav"], dtype="float32")[0] for c in parts]
            if lv != "clean":
                ys = [add_noise(y, float(lv), seed=1000 * k + 10 * j + int(lv)) for j, y in enumerate(ys)]
            audio = np.concatenate(sum([[y, gap] for y in ys], [])[:-1])
            audio = audio / max(1e-6, float(np.abs(audio).max())) * 0.9
            q = V.reference_quality(audio)
            out["refs"][str(lv)] = dict(estimate_snr=q["snr"], seconds=q["seconds"])
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
                y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr, target_sr=R.SR)
                heard = V.transcribe(y16, R.ASR)
                out["takes"].append(dict(level=str(lv), sentence=i, cer=round(V.cer(t["text"], heard), 4),
                                         ecapa=[round(float(v), 5) for v in ecapa(y16)]))
                print("T", person, lv, i, q["snr"], out["takes"][-1]["cer"], flush=True)
        raw[person] = out
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    print("DONE generate")


def analyze(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = json.load(open(os.path.join(raw_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = [p for p in PEOPLE if p in raw]
    A = lambda v: np.asarray(v, np.float64)
    lv = [str(x) for x in LEVELS]
    sim, est, cer = {}, {}, {}
    for p in ps:
        ho = R.unit([r["ecapa"] for r in P[p]["reals"]])
        for l in lv:
            T = [t for t in raw[p]["takes"] if t["level"] == l]
            sim[p, l] = 100 * float(np.mean([A(t["ecapa"]) @ ho for t in T]))
            cer[p, l] = float(np.mean([t["cer"] for t in T]))
            est[p, l] = raw[p]["refs"][l]["estimate_snr"]
    loss = {l: float(np.mean([sim[p, "clean"] - sim[p, l] for p in ps])) for l in lv}
    mest = {l: float(np.mean([est[p, l] for p in ps])) for l in lv}
    ordered = all(all(est[p, a] > est[p, b] for a, b in zip(lv, lv[1:])) for p in ps)
    ok = [l for l in lv[1:] if loss[l] <= WARN_LOSS]
    snr_warn = mest[ok[-1]] if ok else mest["clean"] - 3.0
    bad = [l for l in lv[1:] if loss[l] > BAD_LOSS]
    snr_bad = mest[bad[0]] if bad else mest["10"]
    res = dict(people=ps, loss_points=loss, mean_estimate_snr=mest, ordered_for_every_voice=ordered,
               snr_warn=round(snr_warn, 1) if ordered else None, snr_bad=round(snr_bad, 1) if ordered else None,
               per={f"{p}|{l}": dict(ecapa=sim[p, l], cer=cer[p, l], estimate_snr=est[p, l]) for p in ps for l in lv})
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    L = ["# Reference noise warning: calibration (3 voices)", "",
         "| added noise | estimate_snr (mean) | held-out ECAPA (mean) | loss vs clean | letter error |",
         "|---|---|---|---|---|"]
    for l in lv:
        L.append(f"| {l if l == 'clean' else l + ' dB'} | {mest[l]:.1f} dB | "
                 f"{np.mean([sim[p, l] for p in ps]):.1f} | {loss[l]:+.1f} | "
                 f"{100 * np.mean([cer[p, l] for p in ps]):.1f}% |")
    L += ["", "Per voice estimate_snr (clean / 20 / 15 / 10 dB): " + "; ".join(
        f"{p} " + " / ".join(f"{est[p, l]:.1f}" for l in lv) for p in ps), "",
          f"Estimate orders the levels for every voice: **{ordered}**", "",
          f"**snr_warn = {res['snr_warn']} dB, snr_bad = {res['snr_bad']} dB**" if ordered
          else "**Noise warning not switched on (estimate does not order the levels).**"]
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
