"""Scientific-leakage checks across manifests (run before every training stage).

  * train / valid / test / attacker_train speakers pairwise disjoint (speaker identity is
    compared across corpora that share a speaker space, e.g. LibriTTS-R == LibriSpeech);
  * no excluded speaker (evaluation corpus, known VC targets) in train or valid;
  * no row of a reserved subset (attacker pool, evaluator-ASV training data, test sets)
    in train or valid. A reserved entry may name a shell-style subset pattern ("*",
    "trial-*") and its own "forbidden_splits" (default train + valid), e.g. the frozen SVQ
    evaluation corpus is also kept out of attacker_train.

  python3 training/datasets/leakage.py data/manifests/*.jsonl
"""
import fnmatch
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from datasets.manifest import read  # noqa: E402

EXCLUDED = os.path.join(HERE, "excluded_speakers.json")


def load_exclusions(path=EXCLUDED):
    return json.load(open(path))


def canonical(speaker: str, space: dict) -> str:
    corpus, _, sid = speaker.partition(":")
    return f"{space.get(corpus, corpus)}:{sid}"


DEFAULT_FORBIDDEN_SPLITS = ("train", "valid")


def canonical_corpus(corpus: str, space: dict) -> str:
    return space.get(corpus, corpus)


def reserved_splits(corpus, subset, excl):
    """Splits a row of (corpus, subset) must never enter; empty set = not reserved.
    Corpus names are made canonical (speaker spaces) and subsets match shell-style patterns."""
    space = excl["same_speaker_space"]
    corp = canonical_corpus(corpus, space)
    out = set()
    for r in excl["reserved_subsets"]:
        if canonical_corpus(r["corpus"], space) == corp and fnmatch.fnmatchcase(str(subset or ""), r["subset"]):
            out.update(r.get("forbidden_splits", DEFAULT_FORBIDDEN_SPLITS))
    return out


def check(rows, excl=None):
    excl = excl or load_exclusions()
    space = excl["same_speaker_space"]
    banned = {canonical(s, space) for group in excl["speakers"].values() for s in group}
    errors = []
    by_split = defaultdict(set)
    for i, r in enumerate(rows):
        spk = canonical(r["speaker"], space)
        by_split[r["split"]].add(spk)
        if r["split"] in ("train", "valid") and spk in banned:
            errors.append(f"row {i}: excluded speaker {r['speaker']} in split {r['split']}")
        if r["split"] in reserved_splits(r["corpus"], r.get("subset", ""), excl):
            errors.append(f"row {i}: reserved subset {r['corpus']}/{r.get('subset')} in split {r['split']}")
    splits = sorted(by_split)
    for a in range(len(splits)):
        for b in range(a + 1, len(splits)):
            both = by_split[splits[a]] & by_split[splits[b]]
            if both:
                errors.append(f"speakers in both {splits[a]} and {splits[b]}: {sorted(both)[:5]} ({len(both)})")
    return errors


if __name__ == "__main__":
    rows = [r for p in sys.argv[1:] for r in read(p)]
    errs = check(rows)
    print("\n".join(errs) if errs else "no leakage found")
    sys.exit(1 if errs else 0)
