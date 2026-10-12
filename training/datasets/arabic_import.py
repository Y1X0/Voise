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
import re
import secrets
import shutil
import stat
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


# C9 / R10: import_report.json sits next to the manifests, so it must not carry participant timing
# either. Source file names and unvalidated directory names (often dated, e.g. Telegram-style
# "voice_2026-10-02_14-03-22.wav") are replaced by POSITIONAL references: the entry's index in
# the sorted listing of its parent directory (`file#3` = 4th audio file of that session, sorted
# by name; `dir#0` = first entry). A hash would not hide them: dated names are brute-forceable.
# Ids already in their random schema form (CNS-..., ses-...) are kept. Quoted values inside
# validation messages are replaced by '<value>' (they can echo a precise timestamp).
_CONSENT_ID = re.compile(r"^CNS-[0-9a-f]{12}$")
_SESSION_ID = re.compile(r"^ses-[0-9a-f]{8}$")


def _dir_ref(name, index, pattern):
    return name if pattern.match(name) else f"dir#{index}"


def _scrub(messages):
    return [re.sub(r"'[^']*'", "'<value>'", str(m)) for m in messages]


PROVENANCE_FILE = "provenance.jsonl"
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _nested(a, b):
    return os.path.commonpath([a, b]) == b


def _restricted_dir(out_dir, provenance_dir):
    """Restricted provenance directory, fully separate from the dataset directory (which is what
    training hosts receive): neither inside the other, compared on REAL paths, and not itself a
    symbolic link. Default: a sibling `<out_dir>.provenance`. Created 0700; an existing one is
    reset to 0700 (fails if owned by someone else)."""
    raw = os.path.abspath(provenance_dir or os.path.normpath(os.path.abspath(out_dir)) + ".provenance")
    if os.path.islink(raw):
        raise ValueError("provenance directory must not be a symbolic link")
    pdir, out = os.path.realpath(raw), os.path.realpath(out_dir)
    if _nested(pdir, out) or _nested(out, pdir):
        raise ValueError("provenance directory must be outside the dataset directory (and not contain it)")
    os.makedirs(os.path.dirname(pdir), exist_ok=True)
    try:
        os.mkdir(pdir, 0o700)
    except FileExistsError:
        pass
    if not stat.S_ISDIR(os.lstat(pdir).st_mode):
        raise ValueError("provenance path exists and is not a directory")
    os.chmod(pdir, 0o700)
    return pdir


def _write_restricted(pdir, lines):
    """provenance.jsonl, 0600, written so that no link is ever followed and no partial or
    permissive file is ever visible: directory opened O_NOFOLLOW, temp file created
    O_EXCL|O_NOFOLLOW 0600 (+fchmod), fsync, atomic replace of the directory entry (a symlink at
    the destination is replaced, not followed), fsync of the directory."""
    dfd = os.open(pdir, _DIR_FLAGS)
    tmp = f".{PROVENANCE_FILE}.{secrets.token_hex(8)}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dfd)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            try:
                os.unlink(tmp, dir_fd=dfd)
            except FileNotFoundError:
                pass
            raise
        os.replace(tmp, PROVENANCE_FILE, src_dir_fd=dfd, dst_dir_fd=dfd)
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _provenance_lines(recs, dataset_version):
    return [json.dumps({"recording_id": r["recording_id"], "consent_id": r["consent_id"],
                        "dataset_version": dataset_version, "source_sha256": r["_source_sha256"]},
                       sort_keys=True) + "\n" for r in recs]


def run(incoming, consent_store, out_dir, dataset_version, licence_path="commercial", seed=2026,
        fractions=(0.8, 0.1, 0.1), provenance_dir=None):
    """`provenance_dir`: restricted location for provenance.jsonl (source-file hashes, C9); it is
    kept out of the dataset directory. Default: `<out_dir>.provenance`."""
    import soundfile as sf
    from datasets import build_manifests as BM
    from datasets import consent as CO
    store = CO.load_store(consent_store)
    report = {"dataset_version": dataset_version, "accepted": 0, "rejected": defaultdict(list),
              "duplicates": [], "not_commercial_eligible_speakers": [], "limits": LIMITS}
    recs, fps = [], []
    seen_sha = {}
    for ci, cid in enumerate(sorted(os.listdir(incoming))):
        cdir = os.path.join(incoming, cid)
        if not os.path.isdir(cdir):
            continue
        cref = _dir_ref(cid, ci, _CONSENT_ID)
        rec = store.get(cid)
        if rec is None:
            report["rejected"]["no consent record"].append(cref)
            continue
        if rec["withdrawal"]["status"] != "active":
            report["rejected"]["consent withdrawn"].append(cref)
            continue
        elig = CO.commercial_eligibility(rec)
        if elig != "COMMERCIAL_TRAINING_ELIGIBLE":
            report["not_commercial_eligible_speakers"].append(rec["speaker_pseudonym"])
        for si, sid in enumerate(sorted(os.listdir(cdir))):
            sdir = os.path.join(cdir, sid)
            sref = f"{cref}/{_dir_ref(sid, si, _SESSION_ID)}"
            meta_p = os.path.join(sdir, "session.json")
            if not os.path.isfile(meta_p):
                report["rejected"]["session without session.json"].append(sref)
                continue
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            errs = CO.validate_session(meta)
            if errs or meta["consent_id"] != cid or meta["speaker_id"] != rec["speaker_pseudonym"] or meta["session_id"] != sid:
                report["rejected"]["invalid session metadata"].append(f"{sref}: {_scrub(errs[:3])}")
                continue
            if meta["withdrawal_status"] != "active":
                report["rejected"]["session withdrawn"].append(sref)
                continue
            for fi, name in enumerate(sorted(f for f in os.listdir(sdir) if f.endswith(AUDIO_EXT))):
                where = f"{sref}/file#{fi}"
                src = os.path.join(sdir, name)
                y, info, probs = check_audio(src)
                tpath = os.path.splitext(src)[0] + ".txt"
                text = open(tpath, encoding="utf-8").read().strip() if os.path.exists(tpath) else ""
                norm, tprobs = check_transcript(text, meta["language"], info.get("duration_s", 0.0)) if y is not None else (None, [])
                if probs or tprobs:
                    for p in probs + tprobs:
                        report["rejected"][p.split(":")[0]].append(f"{where}: {_scrub([p])[0]}")
                    continue
                pii = sorted(set(BM.text_privacy_indicators(text)) | set(BM.text_privacy_indicators(norm)))
                if pii:                          # held back before anything is written; kinds only, never text
                    report["rejected"]["transcript needs human review"].append(f"{where}: indicators {pii}")
                    continue
                sha = pcm_sha(y)
                if sha in seen_sha:
                    report["duplicates"].append({"type": "exact", "a": seen_sha[sha], "b": where})
                    continue
                fp = fingerprint(y)
                near = next((r["_where"] for r, f2 in zip(recs, fps)
                             if same_audio(fp, f2)
                             and abs(r["recording"]["duration_s"] - info["duration_s"]) <= LIMITS["near_dup_dur"] * info["duration_s"]), None)
                if near:
                    report["duplicates"].append({"type": "near", "a": near, "b": where})
                    continue
                seen_sha[sha] = where
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
                             "sha256": sha, "_source_sha256": src_sha, "_path": dst, "_where": where})
                fps.append(fp)
    report["accepted"] = len(recs)
    rows = [{"path": r["_path"], "speaker": r["speaker_id"], "session": r["session_id"], "corpus": "own_recordings",
             "subset": dataset_version, "language": r["language"], "dialect": r["dialect"],
             "duration": r["recording"]["duration_s"], "sr": SR, "text": r["transcript_normalized"],
             "consent_id": r["consent_id"], "sha256": r["sha256"], "style": "conversational"} for r in recs]
    os.makedirs(out_dir, exist_ok=True)
    # C9: the hash of the file AS RECEIVED links a training record to the delivered source file
    # (e.g. a Telegram voice note) for anyone holding that file. It is not needed for training,
    # lineage, deletion or withdrawal, so it is kept only in a restricted provenance record
    # outside the dataset directory (data/recording_provenance_schema.json), keyed by consent_id
    # so it can be found and deleted per participant (purge_participant). Written FIRST: if it
    # fails, no recordings.jsonl, manifests or report are written (the audio already written must
    # then be discarded with the failed output directory).
    _write_restricted(_restricted_dir(out_dir, provenance_dir), _provenance_lines(recs, dataset_version))
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


def _replace_text(path, text):
    """Atomic same-directory replace (temp + fsync + rename); keeps no partial file."""
    tmp = f"{path}.{secrets.token_hex(8)}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


PURGE_MARKER = ".purge-in-progress"


def _rewrite_manifests(mdir, rows, index):
    """Rewrite a manifest directory without a window of torn files: build_manifests.write into a
    temp directory beside it (same file system), then os.replace each file into place, the
    index LAST. A crash leaves every file either fully old or fully new; the hashes then
    disagree (verify() refuses the set) until a rerun finishes - never a truncated file.
    The marker PURGE_MARKER (no data in it) is present from before the first replace until after
    the index: it is what allows a rerun to rewrite a set whose hashes disagree. Without it a
    mismatch is someone else's edit and is left for verify() to refuse, never blessed."""
    from datasets import build_manifests as BM
    tmp = f"{mdir}.{secrets.token_hex(6)}.tmp"
    marker = os.path.join(mdir, PURGE_MARKER)
    try:
        BM.write(rows, tmp, index["seed"], index["inputs"], index["dropped"])
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT, 0o600)
        os.fsync(fd)
        os.close(fd)
        names = sorted(n for n in os.listdir(tmp) if n != "manifest_index.json") + ["manifest_index.json"]
        for n in names:
            os.replace(os.path.join(tmp, n), os.path.join(mdir, n))
        os.unlink(marker)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _remove_stale_temps(out, pdir):
    """Temp files/directories left by a HARD crash (no `finally` ran) of an earlier import or
    purge. They can hold any participant's rows (a provenance temp from an interrupted import
    holds every source hash), so a purge removes them first. Returns how many were removed."""
    n = 0
    for name in os.listdir(out):
        p = os.path.join(out, name)
        if os.path.islink(p) or not name.endswith(".tmp"):
            continue
        if name.startswith("manifests.") and os.path.isdir(p):
            shutil.rmtree(p)
            n += 1
        elif os.path.isfile(p):
            os.unlink(p)
            n += 1
    for name in os.listdir(pdir):
        p = os.path.join(pdir, name)
        if name.startswith(f".{PROVENANCE_FILE}.") and name.endswith(".tmp") and os.path.isfile(p) \
                and not os.path.islink(p):
            os.unlink(p)
            n += 1
    return n


def purge_participant(out_dir, consent_id, provenance_dir=None):
    """Remove one participant from an IMPORTED dataset (withdrawal after import): their audio,
    recordings.jsonl rows, manifest rows (manifests rewritten through build_manifests.write, so
    hashes and the leakage gate are recomputed), import_report.json entries (rejections and
    duplicates naming the consent id, the pseudonym in not_commercial_eligible_speakers) and their
    rows in the restricted provenance.jsonl. Idempotent; returns counts only.

    Scope (stated, not claimed beyond it): this output directory and its provenance directory
    only. Backups, copies of the dataset elsewhere (training hosts, Kaggle), trained models and
    the incoming store are NOT touched; the deletion process must handle them separately."""
    from datasets import build_manifests as BM
    if not _CONSENT_ID.match(str(consent_id)):
        raise ValueError("consent id must have the schema form CNS-<12 hex>")
    out = os.path.realpath(out_dir)
    counts = {"recordings": 0, "audio_dirs": 0, "manifest_rows": 0, "report_entries": 0, "provenance_rows": 0}
    pdir = _restricted_dir(out_dir, provenance_dir)
    counts["stale_temps"] = _remove_stale_temps(out, pdir)
    rec_p = os.path.join(out, "recordings.jsonl")
    recs = [json.loads(l) for l in open(rec_p, encoding="utf-8") if l.strip()] if os.path.exists(rec_p) else []
    gone = [r for r in recs if r["consent_id"] == consent_id]
    mdir = os.path.join(out, "manifests")
    index = rows = None
    if os.path.exists(os.path.join(mdir, "manifest_index.json")):
        index = json.load(open(os.path.join(mdir, "manifest_index.json")))
        rows = [json.loads(l) for split in index["manifests"]
                for l in open(os.path.join(mdir, f"{split}.jsonl"), encoding="utf-8") if l.strip()]
    # the speaker comes from recordings.jsonl AND the manifests, and recordings.jsonl is rewritten
    # LAST, so a rerun after a crash at any step still finds everything (idempotent)
    speakers = {r["speaker_id"] for r in gone} | {r["speaker"] for r in rows or [] if r.get("consent_id") == consent_id}
    resuming = rows is not None and os.path.exists(os.path.join(mdir, PURGE_MARKER))
    if rows is not None and not resuming and any(BM._sha(os.path.join(mdir, f"{split}.jsonl")) != digest
                                                 for split, digest in index["manifests"].items()):
        # checked BEFORE anything is deleted, so a refused purge changes nothing
        raise ValueError("manifests do not match manifest_index.json and no purge was in progress: "
                         "refusing to purge (edited by someone else?)")
    audio_root = os.path.join(out, "audio")
    for spk in sorted(speakers):
        d = os.path.join(audio_root, spk.split(":", 1)[1])
        if re.fullmatch(r"spk-[0-9a-f]{10}", os.path.basename(d)) and os.path.isdir(d) and not os.path.islink(d) \
                and os.path.commonpath([os.path.realpath(d), audio_root]) == audio_root:
            shutil.rmtree(d)                         # the pseudonym's directory holds only this participant
            counts["audio_dirs"] += 1
    if rows is not None:
        keep = [r for r in rows if r.get("consent_id") != consent_id and r["speaker"] not in speakers]
        if len(keep) != len(rows) or resuming:
            _rewrite_manifests(mdir, keep, index)
            counts["manifest_rows"] = len(rows) - len(keep)
    rep_p = os.path.join(out, "import_report.json")
    if os.path.exists(rep_p):
        rep = json.load(open(rep_p, encoding="utf-8"))
        n = 0
        for k in list(rep.get("rejected", {})):
            kept = [e for e in rep["rejected"][k] if not str(e).startswith(consent_id + "/") and e != consent_id]
            n += len(rep["rejected"][k]) - len(kept)
            if kept:
                rep["rejected"][k] = kept
            else:
                del rep["rejected"][k]
        dups = [d for d in rep.get("duplicates", []) if not any(str(d.get(x, "")).startswith(consent_id + "/") for x in "ab")]
        n += len(rep.get("duplicates", [])) - len(dups)
        rep["duplicates"] = dups
        nce = [s for s in rep.get("not_commercial_eligible_speakers", []) if s not in speakers]
        n += len(rep.get("not_commercial_eligible_speakers", [])) - len(nce)
        rep["not_commercial_eligible_speakers"] = nce
        if n or gone:
            # derived from what remains, not decremented: a rerun after a crash must not count twice
            # (no purge counter either: an idempotent one would need the consent id, i.e. a link)
            rep["accepted"] = len(recs) - len(gone)
            if os.path.exists(os.path.join(mdir, "manifest_index.json")):
                rep["manifests"] = json.load(open(os.path.join(mdir, "manifest_index.json")))["manifests"]
            _replace_text(rep_p, json.dumps(rep, indent=1, ensure_ascii=False))
        counts["report_entries"] = n
    prov_p = os.path.join(pdir, PROVENANCE_FILE)
    if os.path.exists(prov_p):
        lines = [l for l in open(prov_p, encoding="utf-8") if l.strip()]
        kept_lines = [l for l in lines if json.loads(l)["consent_id"] != consent_id]
        if len(kept_lines) != len(lines):
            _write_restricted(pdir, kept_lines)
            counts["provenance_rows"] = len(lines) - len(kept_lines)
    if gone:
        _replace_text(rec_p, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                     for r in recs if r["consent_id"] != consent_id))
        counts["recordings"] = len(gone)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", required=True)
    ap.add_argument("--consents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--licence-path", choices=["commercial", "research"], default="commercial")
    ap.add_argument("--provenance-dir", help="restricted directory for provenance.jsonl, outside --out "
                                              "(default: <out>.provenance)")
    a = ap.parse_args()
    r = run(a.incoming, a.consents, a.out, a.dataset_version, a.licence_path, provenance_dir=a.provenance_dir)
    print(json.dumps({k: r[k] for k in ("accepted", "manifests")} | {"rejected": {k: len(v) for k, v in r["rejected"].items()},
                                                                       "duplicates": len(r["duplicates"])}, indent=1))


if __name__ == "__main__":
    main()
