"""Dataset manifest: one JSON object per line (JSONL). Audio itself never goes into git.

Required fields
  path        absolute or manifest-relative audio path (wav/flac)
  speaker     corpus-unique speaker id, prefixed with the corpus ("librittsr:1034")
  session     recording session id (chapter, video, call, day) -> cross-session tests
  corpus      corpus name (must have an entry in LICENSES below)
  language    BCP-47 ("en", "ar", "ar-JO", "ar-EG", ...)
  duration    seconds (float)
  sr          sample rate of the file
  split       train | valid | test | attacker_train   (speaker-disjoint, see validate())
Optional
  text, phones, gender (f/m/u), style (read|spontaneous|conversational|broadcast),
  noise (clean|noisy), device, dialect

  python3 training/datasets/manifest.py validate data/manifests/train.jsonl
"""
import json
import sys
from collections import defaultdict

REQUIRED = ("path", "speaker", "session", "corpus", "language", "duration", "sr", "split")
SPLITS = {"train", "valid", "test", "attacker_train"}

# Licence class per corpus as documented in docs/STREAMING_NEURAL_TRAINING_PLAN.md.
# "verify" = terms must be re-checked and accepted by the project owner before use.
LICENSES = {
    "librispeech": "CC-BY-4.0",
    "librittsr": "CC-BY-4.0",
    "vctk": "CC-BY-4.0",
    "commonvoice": "CC0-1.0",
    "ami": "CC-BY-4.0",
    "voxceleb": "verify (research use; audio from YouTube)",
    "fleurs": "CC-BY-4.0",
    "masc": "verify (CC-BY-4.0 reported)",
    "mgb2": "verify (QCRI agreement, research)",
    "qasr": "verify (QCRI agreement, research)",
    "sada": "verify",
    "musan": "CC-BY-4.0",
    "openslr28_rir": "Apache-2.0",
    "own_recordings": "consent forms (project)",
    "example": "n/a (format example only)",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def validate(rows):
    """Returns a list of error strings (empty = valid)."""
    errors = []
    spk_split = defaultdict(set)
    for i, r in enumerate(rows):
        miss = [k for k in REQUIRED if k not in r]
        if miss:
            errors.append(f"row {i}: missing {miss}")
            continue
        if r["split"] not in SPLITS:
            errors.append(f"row {i}: bad split {r['split']!r}")
        if r["corpus"] not in LICENSES:
            errors.append(f"row {i}: corpus {r['corpus']!r} has no licence entry")
        if not str(r["speaker"]).startswith(r["corpus"] + ":"):
            errors.append(f"row {i}: speaker id must be prefixed with '{r['corpus']}:'")
        if not (0 < float(r["duration"]) < 3600):
            errors.append(f"row {i}: implausible duration")
        if int(r["sr"]) < 16000:
            errors.append(f"row {i}: sr {r['sr']} < 16000")
        spk_split[r["speaker"]].add(r["split"])
    for spk, splits in spk_split.items():
        if len(splits) > 1:
            errors.append(f"speaker {spk} appears in several splits {sorted(splits)} (must be disjoint)")
    return errors


def summary(rows):
    out = defaultdict(lambda: {"hours": 0.0, "speakers": set(), "sessions": set()})
    for r in rows:
        k = (r["split"], r["language"])
        out[k]["hours"] += float(r["duration"]) / 3600
        out[k]["speakers"].add(r["speaker"])
        out[k]["sessions"].add((r["speaker"], r["session"]))
    return {f"{s}/{l}": {"hours": round(v["hours"], 2), "speakers": len(v["speakers"]),
                         "sessions": len(v["sessions"])} for (s, l), v in sorted(out.items())}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "validate":
        raise SystemExit("usage: manifest.py validate FILE.jsonl")
    rows = read(sys.argv[2])
    errs = validate(rows)
    print(json.dumps(summary(rows), indent=1))
    if errs:
        print("\n".join(errs[:50]), file=sys.stderr)
        raise SystemExit(1)
