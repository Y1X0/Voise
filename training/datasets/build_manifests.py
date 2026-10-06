#!/usr/bin/env python3
"""Reproducible manifest builder (B4): local corpus layouts -> speaker/session-disjoint
splits -> leakage check (fail-fast) -> hashed JSONL manifests + statistics report.

Nothing is downloaded. Each builder reads an already-downloaded corpus in its official
layout and emits rows in the manifest format (datasets/manifest.py):

  librispeech   <root>/<subset>/<spk>/<chapter>/<spk>-<chapter>-<n>.flac + <spk>-<chapter>.trans.txt
                (also LibriTTS / LibriTTS-R: <utt>.wav + <utt>.normalized.txt); session = chapter
  vctk          <root>/wav48_silence_trimmed/<spk>/<utt>_mic1.flac + <root>/txt/<spk>/<utt>.txt;
                session unknown -> "single" (VCTK speakers cannot be cross-session TEST speakers)
  commonvoice   <root>/<lang>/validated.tsv (client_id, path, sentence) + clips/; session unknown
  segments      a TSV  path<TAB>speaker<TAB>session<TAB>start<TAB>end<TAB>text  (e.g. AMI IHM
                segments derived from the official annotations; session = meeting)
  generic       <root>/<speaker>/<session>/<file>.wav (+ same-name .txt)  (own recordings)

Splitting (deterministic given --seed):
  * excluded speakers / reserved subsets come from datasets/excluded_speakers.json;
  * reserved attacker subsets (e.g. librispeech train-clean-360) -> split attacker_train;
  * remaining speakers are shuffled per corpus and assigned by fractions to train / valid / test;
  * TEST speakers must have >= 2 sessions; their utterances get role=enroll (first session) or
    role=trial (other sessions): session-disjoint enrollment/trial;
  * then datasets/leakage.check + session checks run, and ANY finding aborts (exit 1).
Outputs in --out: train.jsonl valid.jsonl test.jsonl attacker_train.jsonl (sorted),
manifest_index.json (sha256 per manifest, seed, inputs), stats.json, stats.md.
`--verify DIR` re-checks hashes + leakage of an existing directory (exit 1 on any finding).

  python3 training/datasets/build_manifests.py --out data/manifests --seed 2026 \
      --librispeech /data/LibriTTS_R:librittsr:train-clean-100,train-other-500 \
      --librispeech /data/LibriSpeech:librispeech:train-clean-360 \
      --vctk /data/VCTK-Corpus-0.92 --segments /data/ami_ihm_segments.tsv:ami
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from datasets.leakage import canonical, check as leakage_check, load_exclusions  # noqa: E402
from datasets.manifest import LICENSES, validate  # noqa: E402

AUDIO = (".wav", ".flac", ".mp3", ".ogg")


class LeakageError(RuntimeError):
    pass


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _dur(path):
    import soundfile as sf
    i = sf.info(path)
    return i.frames / i.samplerate, i.samplerate


def build_librispeech(root, corpus, subsets, language="en"):
    rows = []
    for sub in subsets:
        for spk_dir in sorted(glob.glob(os.path.join(root, sub, "*"))):
            spk = os.path.basename(spk_dir)
            for ch_dir in sorted(glob.glob(os.path.join(spk_dir, "*"))):
                ch = os.path.basename(ch_dir)
                texts = {}
                for tf in glob.glob(os.path.join(ch_dir, "*.trans.txt")):
                    for line in open(tf, encoding="utf-8"):
                        k, _, t = line.strip().partition(" ")
                        texts[k] = t
                for a in sorted(f for f in glob.glob(os.path.join(ch_dir, "*")) if f.endswith(AUDIO)):
                    key = os.path.splitext(os.path.basename(a))[0]
                    txt = texts.get(key)
                    norm = os.path.splitext(a)[0] + ".normalized.txt"
                    if txt is None and os.path.exists(norm):
                        txt = open(norm, encoding="utf-8").read().strip()
                    d, sr = _dur(a)
                    rows.append({"path": a, "speaker": f"{corpus}:{spk}", "session": ch, "corpus": corpus,
                                 "subset": sub, "language": language, "duration": round(d, 3), "sr": sr,
                                 "text": txt, "style": "read"})
    return rows


def build_vctk(root):
    rows = []
    for a in sorted(glob.glob(os.path.join(root, "wav48_silence_trimmed", "*", "*_mic1.flac"))):
        spk = os.path.basename(os.path.dirname(a))
        utt = os.path.basename(a).replace("_mic1.flac", "")
        tp = os.path.join(root, "txt", spk, utt + ".txt")
        d, sr = _dur(a)
        rows.append({"path": a, "speaker": f"vctk:{spk}", "session": "single", "corpus": "vctk", "subset": "all",
                     "language": "en", "duration": round(d, 3), "sr": sr,
                     "text": open(tp, encoding="utf-8").read().strip() if os.path.exists(tp) else None,
                     "style": "read"})
    return rows


def build_commonvoice(root, lang):
    rows = []
    with open(os.path.join(root, lang, "validated.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a = os.path.join(root, lang, "clips", r["path"])
            if not os.path.exists(a):
                continue
            d, sr = _dur(a)
            rows.append({"path": a, "speaker": f"commonvoice:{r['client_id']}", "session": "unknown",
                         "corpus": "commonvoice", "subset": "validated", "language": lang, "duration": round(d, 3),
                         "sr": sr, "text": r.get("sentence"), "style": "read"})
    return rows


def build_segments(tsv, corpus, language="en"):
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            path, spk, sess, st, en, *txt = line.rstrip("\n").split("\t")
            _, sr = _dur(path)
            rows.append({"path": path, "speaker": f"{corpus}:{spk}", "session": sess, "corpus": corpus,
                         "subset": "segments", "language": language, "offset": float(st),
                         "duration": round(float(en) - float(st), 3), "sr": sr, "text": txt[0] if txt else None,
                         "style": "spontaneous"})
    return rows


def build_generic(root, corpus, language):
    rows = []
    for a in sorted(f for f in glob.glob(os.path.join(root, "*", "*", "*")) if f.endswith(AUDIO)):
        sess = os.path.basename(os.path.dirname(a))
        spk = os.path.basename(os.path.dirname(os.path.dirname(a)))
        tp = os.path.splitext(a)[0] + ".txt"
        d, sr = _dur(a)
        rows.append({"path": a, "speaker": f"{corpus}:{spk}", "session": sess, "corpus": corpus, "subset": "all",
                     "language": language, "duration": round(d, 3), "sr": sr,
                     "text": open(tp, encoding="utf-8").read().strip() if os.path.exists(tp) else None,
                     "style": "conversational"})
    return rows


def assign_splits(rows, seed, fractions=(0.90, 0.05, 0.05), attacker_subsets=(("librispeech", "train-clean-360"),),
                  excl=None):
    """Speaker-disjoint split; TEST speakers need >= 2 sessions and get session-disjoint
    enroll/trial roles. Excluded speakers and reserved non-attacker subsets are dropped."""
    excl = excl or load_exclusions()
    space = excl["same_speaker_space"]
    banned = {canonical(s, space) for g in excl["speakers"].values() for s in g}
    reserved = {(canonical(r["corpus"] + ":x", space).split(":")[0], r["subset"]) for r in excl["reserved_subsets"]}
    attack = {(canonical(c + ":x", space).split(":")[0], s) for c, s in attacker_subsets}
    out, dropped = [], defaultdict(int)
    by_spk = defaultdict(list)
    for r in rows:
        corp = canonical(r["corpus"] + ":x", space).split(":")[0]
        key = (corp, r.get("subset", ""))
        if canonical(r["speaker"], space) in banned:
            dropped["excluded_speaker"] += 1
            continue
        if key in attack:
            out.append(dict(r, split="attacker_train"))
            continue
        if key in reserved or (corp, "*") in reserved:
            dropped["reserved_subset"] += 1
            continue
        by_spk[canonical(r["speaker"], space)].append(r)
    attacker_spk = {canonical(r["speaker"], space) for r in out}
    rng = random.Random(seed)
    by_corpus = defaultdict(list)
    for s in sorted(by_spk):
        if s in attacker_spk:
            dropped["speaker_also_in_attacker_pool"] += len(by_spk[s])
            continue
        by_corpus[s.split(":")[0]].append(s)
    for corp, spks in sorted(by_corpus.items()):
        rng.shuffle(spks)
        n = len(spks)
        n_test = int(round(fractions[2] * n))
        n_valid = int(round(fractions[1] * n))
        multi = [s for s in spks if len({r["session"] for r in by_spk[s]}) >= 2]
        test = multi[:n_test]
        rest = [s for s in spks if s not in set(test)]
        valid, train = rest[:n_valid], rest[n_valid:]
        for split, group in (("train", train), ("valid", valid)):
            for s in group:
                out += [dict(r, split=split) for r in by_spk[s]]
        for s in test:
            sessions = sorted({r["session"] for r in by_spk[s]})
            for r in by_spk[s]:
                out.append(dict(r, split="test", role="enroll" if r["session"] == sessions[0] else "trial"))
    return sorted(out, key=lambda r: (r["split"], r["speaker"], r["session"], r["path"])), dict(dropped)


def session_checks(rows):
    errors = []
    enroll, trial = defaultdict(set), defaultdict(set)
    for r in rows:
        if r["split"] == "test":
            (enroll if r.get("role") == "enroll" else trial)[r["speaker"]].add(r["session"])
    for spk in enroll:
        both = enroll[spk] & trial.get(spk, set())
        if both:
            errors.append(f"test speaker {spk}: session(s) {sorted(both)} in both enroll and trial")
        if not trial.get(spk):
            errors.append(f"test speaker {spk}: no trial session (cross-session test impossible)")
    return errors


def assert_no_leakage(rows):
    errs = validate(rows) + leakage_check(rows) + session_checks(rows)
    if errs:
        raise LeakageError("; ".join(errs[:20]))


def licence_check(rows, path):
    """commercial path: every train/valid row must come from a corpus whose recorded status is
    COMMERCIAL_ALLOWED* (docs/DATASET_LICENSE_MATRIX.md). research path: NOT_ALLOWED only."""
    errs = set()
    for r in rows:
        st = LICENSES.get(r["corpus"], "LICENSE_NOT_VERIFIED")
        if r["split"] in ("train", "valid"):
            if path == "commercial" and not st.startswith("COMMERCIAL_ALLOWED"):
                errs.add(f"corpus {r['corpus']} ({st}) not allowed in the commercial training path")
            if st.startswith("NOT_ALLOWED"):
                errs.add(f"corpus {r['corpus']} is NOT_ALLOWED")
    return sorted(errs)


def stats(rows):
    agg = defaultdict(lambda: {"utterances": 0, "hours": 0.0, "speakers": set(), "sessions": set()})
    for r in rows:
        for key in ((r["split"], "ALL"), (r["split"], r["corpus"]), (r["split"], r["language"])):
            a = agg[key]
            a["utterances"] += 1
            a["hours"] += r["duration"] / 3600
            a["speakers"].add(r["speaker"])
            a["sessions"].add((r["speaker"], r["session"]))
    return {f"{s}/{k}": {"utterances": v["utterances"], "hours": round(v["hours"], 3),
                         "speakers": len(v["speakers"]), "sessions": len(v["sessions"])}
            for (s, k), v in sorted(agg.items())}


def write(rows, out_dir, seed, inputs, dropped):
    os.makedirs(out_dir, exist_ok=True)
    index = {"seed": seed, "inputs": inputs, "dropped": dropped, "manifests": {}}
    for split in ("train", "valid", "test", "attacker_train"):
        p = os.path.join(out_dir, f"{split}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        index["manifests"][split] = _sha(p)
    st = stats(rows)
    json.dump(index, open(os.path.join(out_dir, "manifest_index.json"), "w"), indent=1)
    json.dump(st, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "stats.md"), "w") as f:
        f.write("| split/group | utterances | hours | speakers | sessions |\n|---|---|---|---|---|\n")
        for k, v in st.items():
            f.write(f"| {k} | {v['utterances']} | {v['hours']} | {v['speakers']} | {v['sessions']} |\n")
    return index


def verify(out_dir):
    """Re-check written manifests before training: every sha256 in manifest_index.json must
    match the file on disk and the union must pass the leakage/session checks. Raises
    LeakageError on any mismatch or finding (trainers call this in preflight)."""
    with open(os.path.join(out_dir, "manifest_index.json")) as f:
        index = json.load(f)
    rows = []
    for split, digest in index["manifests"].items():
        p = os.path.join(out_dir, f"{split}.jsonl")
        if _sha(p) != digest:
            raise LeakageError(f"{p}: sha256 differs from manifest_index.json (edited after build)")
        rows += _jsonl(p)
    assert_no_leakage(rows)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", help="verify an existing manifest directory instead of building")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--librispeech", action="append", default=[], help="ROOT:CORPUS:SUBSET[,SUBSET]")
    ap.add_argument("--vctk", action="append", default=[])
    ap.add_argument("--commonvoice", action="append", default=[], help="ROOT:LANG")
    ap.add_argument("--segments", action="append", default=[], help="TSV:CORPUS[:LANG]")
    ap.add_argument("--generic", action="append", default=[], help="ROOT:CORPUS:LANG")
    ap.add_argument("--licence-path", choices=["commercial", "research"], default="commercial",
                    help="research-path models must never ship (docs/DATASET_LICENSE_MATRIX.md)")
    a = ap.parse_args(argv)
    if a.verify:
        try:
            verify(a.verify)
        except LeakageError as e:
            print("LEAKAGE: " + str(e), file=sys.stderr)
            sys.exit(1)
        print("manifests verified: hashes match, no leakage")
        return
    if not a.out:
        ap.error("--out is required when building")
    rows = []
    for spec in a.librispeech:
        root, corpus, subs = spec.rsplit(":", 2)
        rows += build_librispeech(root, corpus, subs.split(","))
    for root in a.vctk:
        rows += build_vctk(root)
    for spec in a.commonvoice:
        root, lang = spec.rsplit(":", 1)
        rows += build_commonvoice(root, lang)
    for spec in a.segments:
        parts = spec.split(":")
        rows += build_segments(parts[0], parts[1], parts[2] if len(parts) > 2 else "en")
    for spec in a.generic:
        root, corpus, lang = spec.rsplit(":", 2)
        rows += build_generic(root, corpus, lang)
    split_rows, dropped = assign_splits(rows, a.seed)
    try:
        assert_no_leakage(split_rows)
    except LeakageError as e:
        print("LEAKAGE: " + str(e), file=sys.stderr)
        sys.exit(1)
    lic = licence_check(split_rows, a.licence_path)
    if lic:
        print("LICENCE: " + "; ".join(lic), file=sys.stderr)
        sys.exit(1)
    idx = write(split_rows, a.out, a.seed, dict(vars(a), licence_path=a.licence_path), dropped)
    print(json.dumps(idx["manifests"], indent=1))


if __name__ == "__main__":
    main()
