#!/usr/bin/env python3
"""Multilingual LibriSpeech (English) integration: ready for when the official source is used.

NOTHING here claims that MLS was downloaded or verified. Status stays COMMERCIAL_PENDING
(datasets/gate.py) until `verify` records BOTH the archive sha256 and the official licence
text sha256 in data/provenance.json.

Steps
  1 download   --url must be copied by the owner from the official page (https://www.openslr.org/94/);
               no URL is built or guessed here. Resumable (HTTP Range), streamed to disk.
  2 checksum   sha256 (always) + md5 (if the official page lists one: --expected-md5) of the archive.
  3 licence    --licence-text: the licence text saved from the official page; it must name
               CC BY 4.0 ("Creative Commons Attribution 4.0" / "CC BY 4.0" / "CC-BY-4.0").
               Recorded (sha256) together with the archive hash -> provenance.json.
  4 manifest   from the extracted official layout:
                 <root>/{train,dev,test}/transcripts.txt   "<spk>_<book>_<seg>\\t<text>"
                 <root>/{train,dev,test}/segments.txt      "<id>\\t<url>\\t<begin>\\t<end>" (durations)
                 <root>/{train,dev,test}/audio/<spk>/<book>/<id>.flac|.opus
  5 speakers   speaker = "mls_en:<spk>" (LibriVox reader id; same speaker space as LibriSpeech)
  6 sessions   session = book id
  7 splits     mode "official": MLS dev -> valid, MLS test -> test (enroll = first book, trial = others),
               train -> train; mode "resplit": datasets/build_manifests.assign_splits.
               Optional speaker-balanced subset of train (cap hours per speaker; --fraction).
  8 leakage    build_manifests.assert_no_leakage (+ excluded/reserved speakers, cross-corpus ids)
  9 statistics build_manifests.write -> hashed manifests + stats.json/stats.md

  python3 training/datasets/mls.py download --url <URL FROM openslr.org/94> --out /data/mls/mls_english.tar.gz
  python3 training/datasets/mls.py verify --archive /data/mls/mls_english.tar.gz --licence-text /data/mls/licence.txt \\
      [--expected-md5 <from official page>]
  python3 training/datasets/mls.py manifest --root /data/mls/mls_english --out data/manifests/mls --fraction 0.10
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROVENANCE = os.path.join(ROOT, "data", "provenance.json")
CORPUS = "mls_en"
LICENCE_PATTERNS = (r"creative\s+commons\s+attribution\s+4\.0", r"\bcc[\s-]?by[\s-]?4\.0\b")


class MlsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- 1 download
def download(url, out_path, chunk=1 << 20, opener=None):
    if not url or not url.startswith("https://"):
        raise MlsError("an https URL copied from the official MLS page is required (no URL is guessed)")
    import urllib.request
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    part = out_path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with (opener or urllib.request.urlopen)(req) as r, open(part, "ab" if have else "wb") as f:
        if have and getattr(r, "status", 206) != 206:          # server ignored Range -> restart
            f.seek(0)
            f.truncate()
        for b in iter(lambda: r.read(chunk), b""):
            f.write(b)
    os.replace(part, out_path)
    return out_path


# ---------------------------------------------------------------------------- 2-3 verify
def file_hashes(path, chunk=1 << 22):
    s, m = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            s.update(b)
            m.update(b)
    return s.hexdigest(), m.hexdigest()


def check_licence_text(text):
    low = text.lower()
    return any(re.search(p, low) for p in LICENCE_PATTERNS)


def verify(archive, licence_text_path, expected_md5=None, expected_sha256=None, source_url=None,
           provenance_path=PROVENANCE, record=True, corpus=CORPUS, licence_patterns=LICENCE_PATTERNS):
    """Record download provenance for `corpus` (default MLS English). Other corpora with an E1
    licence (VCTK, AMI, ...) use the same command with --corpus and their own official licence text."""
    if not os.path.exists(archive):
        raise MlsError(f"archive not found: {archive}")
    if not licence_text_path or not os.path.exists(licence_text_path):
        raise MlsError("the licence text saved from the official MLS page is required (--licence-text)")
    sha, md5 = file_hashes(archive)
    if expected_md5 and md5 != expected_md5.lower():
        raise MlsError(f"md5 mismatch: {md5} != {expected_md5}")
    if expected_sha256 and sha != expected_sha256.lower():
        raise MlsError(f"sha256 mismatch: {sha} != {expected_sha256}")
    with open(licence_text_path, encoding="utf-8", errors="replace") as f:
        lic = f.read()
    if not any(re.search(p, lic.lower()) for p in licence_patterns):
        raise MlsError(f"licence text does not state CC BY 4.0 -> {corpus} stays COMMERCIAL_PENDING")
    if corpus != CORPUS and not source_url:
        raise MlsError("--source-url (the official page the archive came from) is required for non-MLS corpora")
    rec = {"licence_class": "COMMERCIAL_WITH_CONDITIONS", "licence": "CC-BY-4.0" + (" / Public Domain (MLS)" if corpus == CORPUS else ""),
           "source": source_url or "https://www.openslr.org/94/", "archive": os.path.basename(archive),
           "archive_sha256": sha, "archive_md5": md5, "md5_checked_against_official": bool(expected_md5),
           "licence_text_sha256": hashlib.sha256(lic.encode()).hexdigest(),
           "verified_at": datetime.date.today().isoformat()}
    if record:
        prov = {}
        if os.path.exists(provenance_path):
            with open(provenance_path) as f:
                prov = json.load(f)
        prov.setdefault("verified", {})[corpus] = rec
        with open(provenance_path, "w") as f:
            json.dump(prov, f, indent=1)
    return rec


# ---------------------------------------------------------------------------- 4-6 manifest rows
def build_rows(root, splits=("train", "dev", "test")):
    rows = []
    for split in splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        texts, segs = {}, {}
        with open(os.path.join(d, "transcripts.txt"), encoding="utf-8") as f:
            for line in f:
                k, _, t = line.rstrip("\n").partition("\t")
                if k:
                    texts[k] = t
        sp = os.path.join(d, "segments.txt")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4:
                        segs[p[0]] = float(p[3]) - float(p[2])
        for key in sorted(texts):
            spk, book, _ = key.split("_", 2)
            audio = None
            for ext in (".flac", ".opus", ".wav"):
                cand = os.path.join(d, "audio", spk, book, key + ext)
                if os.path.exists(cand):
                    audio = cand
                    break
            if audio is None:
                continue
            dur = segs.get(key)
            if dur is None:
                import soundfile as sf
                i = sf.info(audio)
                dur = i.frames / i.samplerate
            rows.append({"path": audio, "speaker": f"{CORPUS}:{spk}", "session": book, "corpus": CORPUS,
                         "subset": split, "language": "en", "duration": round(dur, 3), "sr": 16000,
                         "text": texts[key], "style": "read", "mls_split": split})
    return rows


def speaker_balanced_subset(rows, fraction, seed=2026):
    """Speaker-balanced subset of the train rows: per-speaker hour cap c with sum(min(h_i, c)) = target.
    Within a speaker, utterances are taken book-interleaved (maximises session diversity)."""
    import random
    train = [r for r in rows if r["mls_split"] == "train"]
    if fraction >= 1.0:
        return rows
    by = defaultdict(list)
    for r in train:
        by[r["speaker"]].append(r)
    hours = {s: sum(r["duration"] for r in v) / 3600 for s, v in by.items()}
    target = fraction * sum(hours.values())
    lo, hi = 0.0, max(hours.values())
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if sum(min(h, c) for h in hours.values()) < target else (lo, c)
    cap = hi
    rng = random.Random(seed)
    keep = []
    for s in sorted(by):
        books = defaultdict(list)
        for r in sorted(by[s], key=lambda r: r["path"]):
            books[r["session"]].append(r)
        order = sorted(books)
        rng.shuffle(order)
        acc, i = 0.0, 0
        queues = [list(books[b]) for b in order]
        while acc < cap * 3600 and any(queues):
            q = queues[i % len(queues)]
            if q:
                r = q.pop(0)
                keep.append(r)
                acc += r["duration"]
            i += 1
    return keep + [r for r in rows if r["mls_split"] != "train"]


def assign_official(rows):
    out = []
    test_books = defaultdict(set)
    for r in rows:
        if r["mls_split"] == "test":
            test_books[r["speaker"]].add(r["session"])
    for r in rows:
        r = dict(r)
        sp = r.pop("mls_split")
        if sp == "train":
            r["split"] = "train"
        elif sp == "dev":
            r["split"] = "valid"
        else:
            if len(test_books[r["speaker"]]) < 2:
                continue                       # cross-session enroll/trial impossible
            r["split"] = "test"
            r["role"] = "enroll" if r["session"] == min(test_books[r["speaker"]]) else "trial"
        out.append(r)
    return sorted(out, key=lambda r: (r["split"], r["speaker"], r["session"], r["path"]))


def make_manifests(root, out_dir, fraction=1.0, mode="official", seed=2026, licence_path="commercial"):
    from datasets import build_manifests as BM
    rows = build_rows(root)
    if not rows:
        raise MlsError(f"no MLS rows under {root}")
    rows = speaker_balanced_subset(rows, fraction, seed)
    if mode == "official":
        split_rows, dropped = assign_official(rows), {}
    else:
        split_rows, dropped = BM.assign_splits([{k: v for k, v in r.items() if k != "mls_split"} for r in rows], seed)
    BM.assert_no_leakage(split_rows)
    lic = BM.licence_check(split_rows, licence_path)
    if lic:
        raise MlsError("dataset gate: " + "; ".join(lic))
    return BM.write(split_rows, out_dir, seed, {"corpus": CORPUS, "root": root, "fraction": fraction, "mode": mode,
                                                "licence_path": licence_path}, dropped)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--url", required=True)
    d.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--archive", required=True)
    v.add_argument("--licence-text", required=True)
    v.add_argument("--expected-md5")
    v.add_argument("--expected-sha256")
    v.add_argument("--source-url")
    v.add_argument("--corpus", default=CORPUS, help="registry id (mls_en, vctk, ami, ...)")
    m = sub.add_parser("manifest")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--fraction", type=float, default=0.10)
    m.add_argument("--mode", choices=["official", "resplit"], default="official")
    m.add_argument("--licence-path", choices=["commercial", "research"], default="commercial")
    a = ap.parse_args()
    try:
        if a.cmd == "download":
            print(download(a.url, a.out))
        elif a.cmd == "verify":
            print(json.dumps(verify(a.archive, a.licence_text, a.expected_md5, a.expected_sha256, a.source_url,
                                    corpus=a.corpus), indent=1))
        else:
            print(json.dumps(make_manifests(a.root, a.out, a.fraction, a.mode, licence_path=a.licence_path)["manifests"], indent=1))
    except MlsError as e:
        print("MLS: " + str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
