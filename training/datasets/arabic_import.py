#!/usr/bin/env python3
"""Importer / validator for consented Arabic recordings (docs/CONSENTED_RECORDING_PLAN.md).

Never collects anything and never assumes consent: every speaker must have a valid record in
the private consent store (data/consent_schema.json), every session a metadata file
(data/recording_session_schema.json). Input layout (as delivered by the collection app):

  <incoming>/<consent_id>/<session_id>/session.json
  <incoming>/<consent_id>/<session_id>/<name>.wav|.flac (+ <name>.txt transcript, UTF-8)

Pipeline (all findings reported in <out>/import_report.json; nothing is silently dropped):
  validate audio      decodable, finite, 0.5-600 s, not silent (RMS > -60 dBFS), clipping <= 1 %
  corruption          decode errors / truncated or empty files -> rejected
  resample/normalise  mono 16 kHz PCM16 FLAC under <out>/audio/<speaker>/<session>/<recording_id>.flac
  duplicates          exact (sha256 of normalised PCM) and near-duplicate (time-aligned log-mel
                      correlation > 0.995 and duration within 2 %) -> rejected, both ids reported
  transcript          present, >= 90 % Arabic/Latin/digit characters, 2-30 normalised chars/s
  consent             record valid, active, in retention, session.consent_id matches, scopes
                      snapshot stored; commercial eligibility per datasets/consent.py
  ids                 speaker_id = consent record pseudonym; session_id from session.json
  manifests           build_manifests: speaker-disjoint train/valid/test (session-disjoint
                      enroll/trial), leakage check (fails on any overlap), strict gate + consent
                      check for the licence path; NOT_COMMERCIAL_TRAINING_ELIGIBLE speakers are
                      excluded from train/valid on the commercial path (reported)

  python3 training/datasets/arabic_import.py --incoming /secure/incoming --consents /secure/consents.jsonl \\
      --out /secure/own_recordings_v1 --dataset-version own-ar-v1 --licence-path commercial
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
SR = 16000
AUDIO_EXT = (".wav", ".flac")
LIMITS = {"min_s": 0.5, "max_s": 600.0, "min_rms_dbfs": -60.0, "max_clip_ratio": 0.01,
          "min_char_ratio": 0.9, "cps_min": 2.0, "cps_max": 30.0, "near_dup_cos": 0.995, "near_dup_dur": 0.02}


def _resample(x, sr):
    if sr == SR:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def check_audio(path):
    """-> (pcm16k float32, info dict, problems list)."""
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        return None, {}, [f"corrupt: undecodable ({type(e).__name__})"]
    probs = []
    if x.size == 0:
        return None, {}, ["corrupt: empty audio"]
    if not np.isfinite(x).all():
        return None, {}, ["corrupt: non-finite samples"]
    info = {"sampling_rate_source_hz": int(sr), "channels_source": int(x.shape[1])}
    mono = x.mean(1)
    clip = float(np.mean(np.abs(x) >= 0.999))
    y = _resample(mono, sr)
    dur = len(y) / SR
    rms = 10 * np.log10(float(np.mean(y.astype(np.float64) ** 2)) + 1e-20)
    info.update({"duration_s": round(dur, 3), "peak_dbfs": round(20 * np.log10(float(np.abs(y).max()) + 1e-12), 2),
                 "rms_dbfs": round(rms, 2), "clipping_ratio": round(clip, 5)})
    if not LIMITS["min_s"] <= dur <= LIMITS["max_s"]:
        probs.append(f"duration: {dur:.2f}s outside [{LIMITS['min_s']}, {LIMITS['max_s']}]")
    if rms < LIMITS["min_rms_dbfs"]:
        probs.append(f"silent: RMS {rms:.1f} dBFS")
    if clip > LIMITS["max_clip_ratio"]:
        probs.append(f"clipping: {clip:.3%}")
    return np.clip(y, -1, 1), info, probs


def check_transcript(text, lang, dur):
    from datasets.ctc_targets import normalize
    if not text or not text.strip():
        return None, ["transcript: missing transcript"]
    raw = [c for c in text if not c.isspace() and not (0x064B <= ord(c) <= 0x0652) and c not in ".,!?؟،؛:-\"'()"]
    ok = sum(1 for c in raw if c.isdigit() or ("؀" <= c <= "ۿ") or ("a" <= c.lower() <= "z"))
    probs = []
    if raw and ok / len(raw) < LIMITS["min_char_ratio"]:
        probs.append(f"transcript: only {ok / len(raw):.0%} Arabic/Latin/digit characters")
    norm = normalize(text, lang)
    cps = len(norm.replace(" ", "")) / max(dur, 1e-3)
    if not norm:
        probs.append("transcript: empty after normalisation")
    elif not LIMITS["cps_min"] <= cps <= LIMITS["cps_max"]:
        probs.append(f"transcript: {cps:.1f} chars/s implausible for {dur:.1f}s audio")
    return norm, probs


def fingerprint(y):
    """Time-aligned 40-band log-mel matrix (50 ms hop), mean/variance normalised. Two recordings
    are near-duplicates when the correlation of their (equal-length-cropped) matrices exceeds
    LIMITS["near_dup_cos"]: re-encodes / re-uploads of the same audio, not similar voices."""
    import librosa
    m = np.log(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=800, n_mels=40) + 1e-6)
    return (m - m.mean()) / (m.std() + 1e-9)


def same_audio(a, b):
    n = min(a.shape[1], b.shape[1])
    if n < 4:
        return False
    x, y = a[:, :n].ravel(), b[:, :n].ravel()
    return float(np.dot(x - x.mean(), y - y.mean()) / (np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()) + 1e-12)) > LIMITS["near_dup_cos"]


def pcm_sha(y):
    return hashlib.sha256((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()).hexdigest()


def run(incoming, consent_store, out_dir, dataset_version, licence_path="commercial", seed=2026,
        fractions=(0.8, 0.1, 0.1)):
    import soundfile as sf
    from datasets import build_manifests as BM
    from datasets import consent as CO
    store = CO.load_store(consent_store)
    report = {"dataset_version": dataset_version, "accepted": 0, "rejected": defaultdict(list),
              "duplicates": [], "not_commercial_eligible_speakers": [], "limits": LIMITS}
    recs, fps = [], []
    seen_sha = {}
    for cid in sorted(os.listdir(incoming)):
        cdir = os.path.join(incoming, cid)
        if not os.path.isdir(cdir):
            continue
        rec = store.get(cid)
        if rec is None:
            report["rejected"]["no consent record"].append(cid)
            continue
        if rec["withdrawal"]["status"] != "active":
            report["rejected"]["consent withdrawn"].append(cid)
            continue
        elig = CO.commercial_eligibility(rec)
        if elig != "COMMERCIAL_TRAINING_ELIGIBLE":
            report["not_commercial_eligible_speakers"].append(rec["speaker_pseudonym"])
        for sid in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, sid)
            meta_p = os.path.join(sdir, "session.json")
            if not os.path.isfile(meta_p):
                report["rejected"]["session without session.json"].append(f"{cid}/{sid}")
                continue
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            errs = CO.validate_session(meta)
            if errs or meta["consent_id"] != cid or meta["speaker_id"] != rec["speaker_pseudonym"] or meta["session_id"] != sid:
                report["rejected"]["invalid session metadata"].append(f"{cid}/{sid}: {errs[:3]}")
                continue
            if meta["withdrawal_status"] != "active":
                report["rejected"]["session withdrawn"].append(f"{cid}/{sid}")
                continue
            for name in sorted(f for f in os.listdir(sdir) if f.endswith(AUDIO_EXT)):
                src = os.path.join(sdir, name)
                y, info, probs = check_audio(src)
                tpath = os.path.splitext(src)[0] + ".txt"
                text = open(tpath, encoding="utf-8").read().strip() if os.path.exists(tpath) else ""
                norm, tprobs = check_transcript(text, meta["language"], info.get("duration_s", 0.0)) if y is not None else (None, [])
                if probs or tprobs:
                    for p in probs + tprobs:
                        report["rejected"][p.split(":")[0]].append(f"{cid}/{sid}/{name}: {p}")
                    continue
                sha = pcm_sha(y)
                if sha in seen_sha:
                    report["duplicates"].append({"type": "exact", "a": seen_sha[sha], "b": f"{cid}/{sid}/{name}"})
                    continue
                fp = fingerprint(y)
                near = next((r["_where"] for r, f2 in zip(recs, fps)
                             if same_audio(fp, f2)
                             and abs(r["recording"]["duration_s"] - info["duration_s"]) <= LIMITS["near_dup_dur"] * info["duration_s"]), None)
                if near:
                    report["duplicates"].append({"type": "near", "a": near, "b": f"{cid}/{sid}/{name}"})
                    continue
                seen_sha[sha] = f"{cid}/{sid}/{name}"
                spk = rec["speaker_pseudonym"]
                rid = "rec-" + sha[:16]
                dst = os.path.join(out_dir, "audio", spk.split(":")[1], sid, rid + ".flac")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                sf.write(dst, y, SR, subtype="PCM_16")
                with open(src, "rb") as f:
                    src_sha = hashlib.sha256(f.read()).hexdigest()
                info.update({"recording_device": meta["recording_device"], "environment": meta["environment"]})
                recs.append({"recording_id": rid, "speaker_id": spk, "session_id": sid, "consent_id": cid,
                             "language": meta["language"], "dialect": meta["dialect"], "transcript": text,
                             "transcript_normalized": norm, "consent_scope": rec["scopes"],
                             "commercial_eligibility": elig, "recording": info, "dataset_version": dataset_version,
                             "sha256": sha, "source_sha256": src_sha, "_path": dst, "_where": f"{cid}/{sid}/{name}"})
                fps.append(fp)
    report["accepted"] = len(recs)
    rows = [{"path": r["_path"], "speaker": r["speaker_id"], "session": r["session_id"], "corpus": "own_recordings",
             "subset": dataset_version, "language": r["language"], "dialect": r["dialect"],
             "duration": r["recording"]["duration_s"], "sr": SR, "text": r["transcript_normalized"],
             "consent_id": r["consent_id"], "sha256": r["sha256"], "style": "conversational"} for r in recs]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "recordings.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False, sort_keys=True) + "\n")
    index = None
    if rows:
        split_rows, dropped = BM.assign_splits(rows, seed, fractions=fractions)
        if licence_path == "commercial":     # non-eligible speakers may still serve as TEST (evaluation consent)
            excl = set(report["not_commercial_eligible_speakers"])
            split_rows = [r for r in split_rows if not (r["speaker"] in excl and r["split"] in ("train", "valid"))]
        BM.assert_no_leakage(split_rows)
        lic = BM.licence_check(split_rows, licence_path, store)
        if lic:
            raise RuntimeError("dataset gate / consent: " + "; ".join(lic))
        index = BM.write(split_rows, os.path.join(out_dir, "manifests"), seed,
                         {"dataset_version": dataset_version, "licence_path": licence_path}, dropped)
    report["rejected"] = dict(report["rejected"])
    report["manifests"] = None if index is None else index["manifests"]
    with open(os.path.join(out_dir, "import_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", required=True)
    ap.add_argument("--consents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--licence-path", choices=["commercial", "research"], default="commercial")
    a = ap.parse_args()
    r = run(a.incoming, a.consents, a.out, a.dataset_version, a.licence_path)
    print(json.dumps({k: r[k] for k in ("accepted", "manifests")} | {"rejected": {k: len(v) for k, v in r["rejected"].items()},
                                                                       "duplicates": len(r["duplicates"])}, indent=1))


if __name__ == "__main__":
    main()
