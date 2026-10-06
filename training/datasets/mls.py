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
  python3 training/datasets/mls.py extract --archive /data/mls/<archive> --dest /data/mls/mls_english
  python3 training/datasets/mls.py extract --archive /data/mls/<archive> --dest /data/mls/mls_english --fraction 0.10
  python3 training/datasets/mls.py manifest --root /data/mls/mls_english --out data/manifests/mls --fraction 0.10

The two `extract` passes (metadata, then only the selected 10 % train audio + dev/test) avoid
extracting the other ~90 % of the corpus; `manifest` selects the same subset from the metadata, so
it gives the same result on a selective or a full extraction (tested).
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


# ---------------------------------------------------------------------------- 3b selective extraction
SPLIT_DIRS = ("train", "dev", "test")
META_FILES = ("transcripts.txt", "segments.txt")
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def extract(archive, dest, fraction=None, seed=2026):
    """Stream the (verified) archive once and extract only what is needed; nothing is extracted twice.
      fraction None : metadata only ({train,dev,test}/{transcripts,segments}.txt)
      fraction f    : + audio of the select_train_keys(dest, f) train utterances + all dev/test audio
    Member paths are never used as given: only <split>/<metadata> and <split>/audio/<spk>/<book>/<key>.<ext>
    with safe components are written, under `dest`. Resumable (complete files are skipped)."""
    import shutil
    import tarfile
    sel = None
    if fraction is not None:
        sel = set(select_train_keys(dest, fraction, seed))
    n = {"metadata": 0, "audio": 0, "skipped_existing": 0, "bytes_written": 0}
    with tarfile.open(archive, "r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            i = next((j for j, c in enumerate(parts) if c in SPLIT_DIRS), None)
            if i is None:
                continue
            rel = parts[i:]
            if not all(_SAFE.match(c) and c not in (".", "..") for c in rel):
                continue
            if len(rel) == 2 and rel[1] in META_FILES:
                kind = "metadata"
            elif sel is not None and len(rel) == 5 and rel[1] == "audio":
                stem, ext = os.path.splitext(rel[4])
                if ext not in (".flac", ".opus", ".wav") or not stem.startswith(f"{rel[2]}_{rel[3]}_"):
                    continue
                if rel[0] == "train" and stem not in sel:
                    continue
                kind = "audio"
            else:
                continue
            out = os.path.join(dest, *rel)
            if os.path.exists(out) and os.path.getsize(out) == m.size:
                n["skipped_existing"] += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            src = t.extractfile(m)
            with open(out + ".part", "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            os.replace(out + ".part", out)
            n[kind] += 1
            n["bytes_written"] += m.size
    rep = dict(n, archive=os.path.basename(archive), fraction=fraction, seed=seed,
               selected_train_utterances=None if sel is None else len(sel))
    with open(os.path.join(dest, "extract_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


# ---------------------------------------------------------------------------- 4-6 manifest rows
def build_rows(root, splits=("train", "dev", "test"), keep=None):
    """Manifest rows from the official layout. `keep`: {split: ordered list of keys} restricts a split
    to those keys (texts and durations of other keys are never held in memory) and is STRICT: a kept
    key without audio raises (the selection was made on metadata; nothing is silently dropped).
    Without `keep`, keys without audio are skipped (previous behaviour)."""
    rows = []
    for split in splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        want = None if not keep or split not in keep else set(keep[split])
        texts, segs = {}, {}
        with open(os.path.join(d, "transcripts.txt"), encoding="utf-8") as f:
            for line in f:
                k, _, t = line.rstrip("\n").partition("\t")
                if k and (want is None or k in want):
                    texts[k] = t
        sp = os.path.join(d, "segments.txt")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4 and (want is None or p[0] in want):
                        segs[p[0]] = float(p[3]) - float(p[2])
        if want is not None and len(texts) != len(want):
            raise MlsError(f"{split}: {len(want) - len(texts)} selected keys have no transcript")
        missing = 0
        for key in sorted(texts):
            spk, book, _ = key.split("_", 2)
            audio = None
            for ext in (".flac", ".opus", ".wav"):
                cand = os.path.join(d, "audio", spk, book, key + ext)
                if os.path.exists(cand):
                    audio = cand
                    break
            if audio is None:
                missing += 1
                continue
            dur = segs.get(key)
            if dur is None:
                import soundfile as sf
                i = sf.info(audio)
                dur = i.frames / i.samplerate
            rows.append({"path": audio, "speaker": f"{CORPUS}:{spk}", "session": book, "corpus": CORPUS,
                         "subset": split, "language": "en", "duration": round(dur, 3), "sr": 16000,
                         "text": texts[key], "style": "read", "mls_split": split})
        if want is not None and missing:
            raise MlsError(f"{split}: {missing} selected utterances have no audio under {d}/audio "
                           f"(extract them: mls.py extract --fraction ...)")
    return rows


def _train_metadata(root):
    """Train keys (sorted) and durations (seconds, rounded to ms exactly as build_rows does) from
    transcripts.txt + segments.txt only, in compact arrays (no texts, no audio access).
    RAM ~ 10.8 M keys x (~40 B key + 8 B) for MLS English instead of ~1 KB per full row dict."""
    import numpy as np
    d = os.path.join(root, "train")

    def keys_of(path, col_dur=False):
        ks, ds, buf, dbuf = [], [], [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if not p[0] or (col_dur and len(p) < 4):
                    continue
                buf.append(p[0].encode())
                if col_dur:
                    dbuf.append(round(float(p[3]) - float(p[2]), 3))
                if len(buf) == 1_000_000:
                    ks.append(np.array(buf)); buf = []
                    if col_dur:
                        ds.append(np.array(dbuf, np.float64)); dbuf = []
        if buf:
            ks.append(np.array(buf))
            if col_dur:
                ds.append(np.array(dbuf, np.float64))
        w = max([a.dtype.itemsize for a in ks] or [1])
        k = np.concatenate([a.astype(f"S{w}") for a in ks]) if ks else np.array([], "S1")
        return k, (np.concatenate(ds) if col_dur and ds else np.array([], np.float64))
    tk, _ = keys_of(os.path.join(d, "transcripts.txt"))
    tk = np.unique(tk)                                         # sorted (bytes order == str order for ASCII ids)
    sp = os.path.join(d, "segments.txt")
    if not os.path.exists(sp):
        raise MlsError(f"{sp} missing: the metadata-only selection needs official segment durations")
    sk, sd = keys_of(sp, col_dur=True)
    o = np.argsort(sk, kind="stable")
    sk, sd = sk[o], sd[o]
    i = np.clip(np.searchsorted(sk, tk, side="right") - 1, 0, max(len(sk) - 1, 0))   # last duplicate wins, as in a dict
    if len(tk) and (not len(sk) or not np.all(sk[i] == tk)):
        raise MlsError("train keys without a duration in segments.txt")
    return tk, sd[i]


def select_train_keys(root, fraction, seed=2026):
    """Same subset as speaker_balanced_subset(build_rows(root), fraction, seed) (tested), computed from
    the metadata only, so the 10 % can be chosen BEFORE any audio is extracted. Returns the selected
    train keys in the order speaker_balanced_subset emits them."""
    import random
    tk, dur = _train_metadata(root)
    blocks, hours = {}, {}                                     # speaker -> (start, end); insertion order kept
    cur, start, acc = None, 0, 0.0
    for j in range(len(tk)):
        spk = tk[j].split(b"_", 1)[0]
        if spk != cur:
            if cur is not None:
                blocks[cur], hours[cur] = (start, j), acc / 3600
            cur, start, acc = spk, j, 0
        acc += float(dur[j])
    if cur is not None:
        blocks[cur], hours[cur] = (start, len(tk)), acc / 3600
    if fraction >= 1.0:
        return [k.decode() for k in tk]
    target = fraction * sum(hours.values())
    lo, hi = 0.0, max(hours.values())
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if sum(min(h, c) for h in hours.values()) < target else (lo, c)
    cap = hi
    rng = random.Random(seed)
    keep = []
    for spk in sorted(blocks, key=lambda b: f"{CORPUS}:{b.decode()}"):
        a, b = blocks[spk]
        books = defaultdict(list)
        for j in range(a, b):
            books[tk[j].split(b"_", 2)[1].decode()].append(j)
        order = sorted(books)
        rng.shuffle(order)
        acc, i = 0.0, 0
        queues = [list(books[bk]) for bk in order]
        while acc < cap * 3600 and any(queues):
            q = queues[i % len(queues)]
            if q:
                j = q.pop(0)
                keep.append(tk[j].decode())
                acc += float(dur[j])
            i += 1
    return keep


def speaker_balanced_subset(rows, fraction, seed=2026):
    """Speaker-balanced subset of the train rows: per-speaker hour cap c with sum(min(h_i, c)) = target.
    Within a speaker, utterances are taken book-interleaved (maximises session diversity).
    Reference implementation on full rows; make_manifests uses select_train_keys (same result,
    metadata only, bounded memory)."""
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
    if fraction >= 1.0:
        rows = build_rows(root)
    else:
        sel = select_train_keys(root, fraction, seed)          # metadata only; same subset as before
        rows = build_rows(root, keep={"train": sel})
        pos = {k: i for i, k in enumerate(sel)}
        tr = sorted((r for r in rows if r["mls_split"] == "train"),
                    key=lambda r: pos[os.path.splitext(os.path.basename(r["path"]))[0]])
        rows = tr + [r for r in rows if r["mls_split"] != "train"]
    if not rows:
        raise MlsError(f"no MLS rows under {root}")
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
    x = sub.add_parser("extract", help="stream the archive; metadata only, or + the selected audio (--fraction)")
    x.add_argument("--archive", required=True)
    x.add_argument("--dest", required=True)
    x.add_argument("--fraction", type=float, help="extract the audio of this speaker-balanced train subset + dev/test")
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
        elif a.cmd == "extract":
            print(json.dumps(extract(a.archive, a.dest, a.fraction), indent=1))
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
