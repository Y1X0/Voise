#!/usr/bin/env python3
"""Speaker-anonymization experiments (A: baseline presets, B: single factors,
C: combinations, S: per-session randomisation) on the REAL multi-speaker corpus
built by scripts/build_eval_corpus.py.

For every configuration each utterance is streamed through the real engine
(dsp/build/voiceanon_eval --render-only, 192-sample blocks, 16 kHz) and scored:

  speaker similarity  Resemblyzer GE2E d-vectors (evaluation tool only)
                      distributions for same/different speaker x original/processed
  linkage             verification ROC-AUC / EER (speaker-level bootstrap 95 % CI)
                      for "ignorant" (enrol original, test processed) and
                      "lazy-informed" (enrol processed, test processed) attackers,
                      plus closed-set identification top-1 accuracy
  intelligibility     pocketsphinx WER of processed vs. ASR of the original
                      (relative WER: content preservation proxy), ESTOI (pystoi)
  naturalness         DNSMOS P.835 OVRL / SIG (speechmos; non-intrusive proxy)
  latency / CPU       engine latency and real-time factor reported by the renderer

usage: anon_experiments.py [--configs A_balanced,B1,...] [--boot 1000]
Writes eval-experiments/results.json and prints Markdown tables.
"""
import argparse
import collections
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "eval-corpus")
OUT = os.path.join(ROOT, "eval-experiments")
EVAL = os.path.join(ROOT, "dsp", "build", "voiceanon_eval")

# "neutral" = Balanced preset with every anonymization dimension switched off,
# so single-factor experiments change exactly one thing.
NEUTRAL = ["--preset", "balanced", "--pitch", "0", "--formant", "0", "--tilt", "0", "--intonation", "1"]


def cfg(cid, group, desc, args, session_random=False):
    return {"id": cid, "group": group, "desc": desc, "args": args, "session_random": session_random}


CONFIGS = [
    # A - frozen baseline presets (unchanged app behaviour)
    cfg("A_natural", "A", "Natural preset", ["--preset", "natural"]),
    cfg("A_balanced", "A", "Balanced preset", ["--preset", "balanced"]),
    cfg("A_strong", "A", "Strong preset", ["--preset", "strong"]),
    # B - single factors (one parameter each)
    cfg("B0_neutral", "B", "chain only (NS/EQ/AGC), no anonymization", NEUTRAL),
    cfg("B1_formant10", "B", "formant shift 10.6 %", NEUTRAL[:-6] + ["--formant", "10.6", "--tilt", "0", "--intonation", "1"]),
    cfg("B1_formant20", "B", "formant shift 20 %", NEUTRAL[:-6] + ["--formant", "20", "--tilt", "0", "--intonation", "1"]),
    cfg("B2_reshape6", "B", "spectral-envelope reshape +-6 dB", NEUTRAL + ["--reshape", "6"]),
    cfg("B2_reshape10", "B", "spectral-envelope reshape +-10 dB", NEUTRAL + ["--reshape", "10"]),
    cfg("B3_tilt3", "B", "spectral tilt 3 dB", NEUTRAL[:-4] + ["--tilt", "3", "--intonation", "1"]),
    cfg("B4_pitch3", "B", "pitch shift 2.93 st", ["--preset", "balanced", "--formant", "0", "--tilt", "0", "--intonation", "1"]),
    cfg("B4_pitch45", "B", "pitch shift 4.5 st", ["--preset", "balanced", "--pitch", "4.5", "--formant", "0", "--tilt", "0", "--intonation", "1"]),
    cfg("B5_prosody", "B", "controlled prosody: intonation x0.75 + drift 1 st", NEUTRAL[:-2] + ["--intonation", "0.75", "--drift", "1"]),
    cfg("B6_micro", "B", "temporal micro-variation: formant jitter 5 %", NEUTRAL + ["--fjitter", "5"]),
    cfg("B7_energy", "B", "energy-contour flattening 1.0", NEUTRAL + ["--flatten", "1"]),
    # C - combinations
    cfg("C1_form_resh", "C", "formant 16 % + reshape 6 dB", NEUTRAL[:-6] + ["--formant", "16", "--tilt", "0", "--intonation", "1", "--reshape", "6"]),
    cfg("C2_pitch_form", "C", "pitch 2.93 st + formant 16 %", ["--preset", "balanced", "--formant", "16", "--tilt", "0", "--intonation", "1"]),
    cfg("C3_p_f_resh", "C", "pitch 2.93 + formant 16 % + reshape 6 dB", ["--preset", "balanced", "--formant", "16", "--tilt", "0", "--intonation", "1", "--reshape", "6"]),
    cfg("C4_p_f_r_pros", "C", "C3 + intonation x0.75 + drift 1 st", ["--preset", "balanced", "--formant", "16", "--tilt", "0", "--intonation", "0.75", "--reshape", "6", "--drift", "1"]),
    cfg("C5_all", "C", "C4 + formant jitter 4 % + flatten 0.5 + tilt 2", ["--preset", "balanced", "--formant", "16", "--tilt", "2", "--intonation", "0.75", "--reshape", "6", "--drift", "1", "--fjitter", "4", "--flatten", "0.5"]),
    cfg("C6_max", "C", "pitch 4.5 + formant 20 % + reshape 10 + tilt 3", ["--preset", "balanced", "--pitch", "4.5", "--formant", "20", "--tilt", "3", "--intonation", "0.75", "--reshape", "10"]),
    # S - per-session randomisation (new random seed + +-10 % magnitudes per utterance = per session)
    cfg("S_balanced", "S", "Balanced, randomised per session", ["--preset", "balanced"], True),
    cfg("S_C3", "S", "C3, randomised per session (reshape curve differs per session)",
        ["--preset", "balanced", "--formant", "16", "--tilt", "0", "--intonation", "1", "--reshape", "6"], True),
]


def utterances():
    files = sorted(glob.glob(os.path.join(CORPUS, "*.wav")))
    return [(os.path.basename(f)[:-4], f) for f in files]


def speaker(u):
    return u.split("__")[0]


def speaker_directions(utts):
    """Auto-direction emulation: speakers with median F0 < 165 Hz go up, others down
    (the app remembers this between sessions, so it applies from the start)."""
    import librosa
    f0s = collections.defaultdict(list)
    for name, path in utts:
        x, sr = sf.read(path)
        f0 = librosa.yin(x.astype(np.float32), fmin=65, fmax=500, sr=sr, frame_length=1024)
        rms = librosa.feature.rms(y=x.astype(np.float32), frame_length=1024, hop_length=256)[0]
        n = min(len(f0), len(rms))
        voiced = f0[:n][rms[:n] > 0.3 * np.max(rms)]
        f0s[speaker(name)] += list(voiced)
    return {s: ("up" if np.median(v) < 165 else "down") for s, v in f0s.items()}, \
           {s: float(np.median(v)) for s, v in f0s.items()}


def scale_args(args, factor):
    out, i = [], 0
    while i < len(args):
        if args[i] in ("--pitch", "--formant", "--reshape") and i + 1 < len(args):
            out += [args[i], f"{float(args[i + 1]) * factor:.3f}"]
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def render(config, utts, dirs):
    d = os.path.join(OUT, config["id"])
    os.makedirs(d, exist_ok=True)
    rtf, lat = [], []
    rng = random.Random(1234)
    for idx, (name, path) in enumerate(utts):
        args = list(config["args"])
        if config["session_random"]:
            # Emulates the app's per-session variation (and a new reshape seed per session).
            factor = 0.9 + 0.2 * rng.random()
            if "--pitch" not in args:
                args += ["--pitch", "2.925"]
            args = scale_args(args, factor) + ["--seed", str(1000 + idx)]
        res = subprocess.run([EVAL, *args, "--direction", dirs[speaker(name)], "--render-only",
                              "--tag", "p", "--out", d, path], capture_output=True, text=True, check=True)
        info = json.loads(res.stdout.strip().splitlines()[-1])
        rtf.append(info["rtf"])
        lat.append(info["latencyMs"])
    return d, float(np.mean(rtf)), float(np.mean(lat))


# --------------------------------------------------------------------------
def embed_all(paths):
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder(device="cpu", verbose=False)
    out = {}
    for key, p in paths.items():
        e = enc.embed_utterance(preprocess_wav(p))
        out[key] = e / np.linalg.norm(e)
    return out


def roc_auc_eer(targets, nontargets):
    t, n = np.asarray(targets), np.asarray(nontargets)
    if len(t) == 0 or len(n) == 0:
        return float("nan"), float("nan")
    # AUC = P(target > non-target), ties count half (Mann-Whitney).
    allv = np.concatenate([t, n])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    auc = (ranks[:len(t)].sum() - len(t) * (len(t) + 1) / 2) / (len(t) * len(n))
    thr = np.sort(allv)
    frr = np.array([(t < x).mean() for x in thr])
    far = np.array([(n >= x).mean() for x in thr])
    i = int(np.argmin(np.abs(frr - far)))
    return float(auc), float((frr[i] + far[i]) / 2)


def pair_lists(names, emb_a, emb_b, symmetric):
    """same/diff speaker score lists; pairs carry speaker ids for bootstrapping."""
    same, diff = [], []
    for i, u in enumerate(names):
        for j, v in enumerate(names):
            if i == j or (symmetric and j < i):
                continue
            s = float(np.dot(emb_a[u], emb_b[v]))
            (same if speaker(u) == speaker(v) else diff).append((speaker(u), speaker(v), s))
    return same, diff


def bootstrap(same, diff, n_boot, seed=7):
    spks = sorted({a for a, _, _ in same} | {a for a, _, _ in diff})
    rng = np.random.default_rng(seed)
    aucs, eers = [], []
    for _ in range(n_boot):
        pick = collections.Counter(rng.choice(spks, size=len(spks), replace=True))
        t = [s for a, b, s in same for _ in range(pick[a])]
        nn = [s for a, b, s in diff if pick[a] and pick[b] for _ in range(pick[a] * pick[b])]
        if t and nn:
            a, e = roc_auc_eer(t, nn)
            aucs.append(a)
            eers.append(e)
    q = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    return q(aucs), q(eers)


def identification(names, enrol, cand):
    """Closed-set: for each enrolment utterance pick the speaker whose (other)
    candidate utterance scores highest."""
    correct = 0
    for u in names:
        best, best_s = None, -9
        for v in names:
            if v == u:
                continue
            s = float(np.dot(enrol[u], cand[v]))
            if s > best_s:
                best, best_s = speaker(v), s
        correct += best == speaker(u)
    return correct / len(names)


def stats(xs):
    xs = np.asarray([x[2] for x in xs])
    return {"mean": float(xs.mean()), "median": float(np.median(xs)), "std": float(xs.std(ddof=1)),
            "min": float(xs.min()), "max": float(xs.max()), "n": int(len(xs)),
            "hist": np.histogram(xs, bins=10, range=(0, 1))[0].tolist()}


# --------------------------------------------------------------------------
def asr(paths):
    from pocketsphinx import Decoder
    dec = Decoder(samprate=16000)
    out = {}
    for key, p in paths.items():
        x, _ = sf.read(p)
        dec.start_utt()
        dec.process_raw((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(), full_utt=True)
        dec.end_utt()
        out[key] = dec.hyp().hypstr.split() if dec.hyp() else []
    return out


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def estoi(orig, proc):
    from pystoi import stoi
    vals = []
    for k in orig:
        a, sr = sf.read(orig[k])
        b, _ = sf.read(proc[k])
        n = min(len(a), len(b))
        vals.append(stoi(a[:n], b[:n], sr, extended=True))
    return float(np.mean(vals))


def dnsmos(paths):
    from speechmos import dnsmos
    ovrl, sig = [], []
    for p in paths.values():
        x, sr = sf.read(p)
        r = dnsmos.run(np.clip(x, -1, 1).astype(np.float32), sr)
        ovrl.append(r["ovrl_mos"])
        sig.append(r["sig_mos"])
    return float(np.mean(ovrl)), float(np.std(ovrl)), float(np.mean(sig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    utts = utterances()
    names = [n for n, _ in utts]
    os.makedirs(OUT, exist_ok=True)
    dirs, medf0 = speaker_directions(utts)
    configs = [c for c in CONFIGS if not args.configs or c["id"] in args.configs.split(",")]

    orig_paths = {n: p for n, p in utts}
    print(f"corpus: {len(set(map(speaker, names)))} speakers, {len(names)} utterances", file=sys.stderr)
    emb_o = embed_all(orig_paths)
    asr_o = asr(orig_paths)
    mos_o = dnsmos(orig_paths)
    same_oo, diff_oo = pair_lists(names, emb_o, emb_o, True)
    auc_oo, eer_oo = roc_auc_eer([s for *_, s in same_oo], [s for *_, s in diff_oo])
    ci_oo = bootstrap(same_oo, diff_oo, args.boot)
    results = {"corpus": {"speakers": len(set(map(speaker, names))), "utterances": len(names),
                          "directions": dirs, "medianF0": medf0},
               "original": {"same": stats(same_oo), "diff": stats(diff_oo), "auc": auc_oo, "eer": eer_oo,
                            "auc_ci": ci_oo[0], "eer_ci": ci_oo[1],
                            "ident_top1": identification(names, emb_o, emb_o),
                            "dnsmos_ovrl": mos_o[0], "dnsmos_sig": mos_o[2]},
               "configs": {}}

    for c in configs:
        print(f"[{c['id']}] {c['desc']}", file=sys.stderr)
        d, rtf, lat = render(c, utts, dirs)
        proc_paths = {n: os.path.join(d, f"{n}_p.wav") for n in names}
        emb_p = embed_all(proc_paths)
        same_op, diff_op = pair_lists(names, emb_o, emb_p, False)   # ignorant attacker
        same_pp, diff_pp = pair_lists(names, emb_p, emb_p, True)    # lazy-informed attacker
        auc_i, eer_i = roc_auc_eer([s for *_, s in same_op], [s for *_, s in diff_op])
        auc_l, eer_l = roc_auc_eer([s for *_, s in same_pp], [s for *_, s in diff_pp])
        ci_i, ci_l = bootstrap(same_op, diff_op, args.boot), bootstrap(same_pp, diff_pp, args.boot)
        m_same_op = np.mean([s for *_, s in same_op])
        m_diff_op = np.mean([s for *_, s in diff_op])
        residual = (m_same_op - m_diff_op) / (results["original"]["same"]["mean"] - results["original"]["diff"]["mean"])
        asr_p = asr(proc_paths)
        edits = sum(edit_distance(asr_o[n], asr_p[n]) for n in names)
        words = sum(len(asr_o[n]) for n in names)
        mos = dnsmos(proc_paths)
        results["configs"][c["id"]] = {
            "group": c["group"], "desc": c["desc"], "args": c["args"], "sessionRandom": c["session_random"],
            "same_orig_proc": stats(same_op), "diff_orig_proc": stats(diff_op),
            "same_proc_proc": stats(same_pp), "diff_proc_proc": stats(diff_pp),
            "residual_identity": float(residual),
            "ignorant": {"auc": auc_i, "eer": eer_i, "auc_ci": ci_i[0], "eer_ci": ci_i[1],
                         "ident_top1": identification(names, emb_o, emb_p)},
            "lazy": {"auc": auc_l, "eer": eer_l, "auc_ci": ci_l[0], "eer_ci": ci_l[1],
                     "ident_top1": identification(names, emb_p, emb_p)},
            "rel_wer": edits / max(1, words), "estoi": estoi(orig_paths, proc_paths),
            "dnsmos_ovrl": mos[0], "dnsmos_ovrl_std": mos[1], "dnsmos_sig": mos[2],
            "rtf": rtf, "latencyMs": lat,
        }
        json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print_tables(results)


def print_tables(r):
    o = r["original"]
    print(f"Corpus: {r['corpus']['speakers']} speakers, {r['corpus']['utterances']} utterances (real speech)\n")
    print(f"Original: same-speaker {o['same']['mean']:.3f} (median {o['same']['median']:.3f}, sd {o['same']['std']:.3f}), "
          f"different-speaker {o['diff']['mean']:.3f} (median {o['diff']['median']:.3f}, sd {o['diff']['std']:.3f}); "
          f"EER {100*o['eer']:.1f} % [{100*o['eer_ci'][0]:.1f}-{100*o['eer_ci'][1]:.1f}], AUC {o['auc']:.3f}, "
          f"top-1 {100*o['ident_top1']:.0f} %, DNSMOS OVRL {o['dnsmos_ovrl']:.2f}\n")
    print("| config | same orig→proc mean / median / sd / max | same proc↔proc mean | diff proc↔proc mean | residual identity |")
    print("|---|---|---|---|---|")
    for k, c in r["configs"].items():
        s, p, dp = c["same_orig_proc"], c["same_proc_proc"], c["diff_proc_proc"]
        print(f"| {k} | {s['mean']:.3f} / {s['median']:.3f} / {s['std']:.3f} / {s['max']:.3f} | {p['mean']:.3f} | "
              f"{dp['mean']:.3f} | {c['residual_identity']:.2f} |")
    print("\n| config | ignorant EER [95 % CI] | ignorant AUC | ignorant top-1 | lazy EER [95 % CI] | lazy top-1 | rel. WER | ESTOI | DNSMOS OVRL | RTF |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, c in r["configs"].items():
        i, l = c["ignorant"], c["lazy"]
        print(f"| {k} | {100*i['eer']:.1f} % [{100*i['eer_ci'][0]:.0f}-{100*i['eer_ci'][1]:.0f}] | {i['auc']:.3f} | "
              f"{100*i['ident_top1']:.0f} % | {100*l['eer']:.1f} % [{100*l['eer_ci'][0]:.0f}-{100*l['eer_ci'][1]:.0f}] | "
              f"{100*l['ident_top1']:.0f} % | {100*c['rel_wer']:.0f} % | {c['estoi']:.2f} | {c['dnsmos_ovrl']:.2f} | {c['rtf']:.4f} |")


if __name__ == "__main__":
    main()
