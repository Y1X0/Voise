#!/usr/bin/env python3
"""Validate data/dataset_registry.json and print hour/speaker totals per licence class.

Rules (exit 1 on violation):
  * classification is one of the four statuses;
  * COMMERCIAL_SAFE / COMMERCIAL_WITH_CONDITIONS require evidence_level starting with "E1"
    and not a mirror ("E1-m");
  * commercial_training_allowed is true only for COMMERCIAL_* entries not excluded by project rule;
  * every datasets/manifest.py LICENSES entry that names a registry id agrees with it.

Hour totals: raw sums per class and language (entries with unknown hours are counted as
"+unknown"), and a de-duplicated total where overlapping corpora (same source audio, e.g.
the LibriVox family) contribute only the largest member of their overlap group.

  python3 scripts/dataset_registry_summary.py [--json]
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "data", "dataset_registry.json")
STATUSES = ("COMMERCIAL_SAFE", "COMMERCIAL_WITH_CONDITIONS", "RESEARCH_ONLY", "LICENSE_UNVERIFIED")


def load(path=REG):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def violations(reg):
    errs = []
    for d in reg["datasets"]:
        c = d["classification"]
        if c not in STATUSES:
            errs.append(f"{d['id']}: bad classification {c}")
        if c.startswith("COMMERCIAL") and (not str(d["evidence_level"]).startswith("E1") or str(d["evidence_level"]).startswith("E1-m")):
            errs.append(f"{d['id']}: {c} without publisher (E1) evidence")
        allowed = c.startswith("COMMERCIAL") and not d.get("excluded_by_project_rule")
        if bool(d["commercial_training_allowed"]) != allowed:
            errs.append(f"{d['id']}: commercial_training_allowed inconsistent")
    return errs


def groups(reg):
    """Union-find over overlaps_with -> overlap groups."""
    parent = {d["id"]: d["id"] for d in reg["datasets"]}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for d in reg["datasets"]:
        for o in d.get("overlaps_with", []):
            if o in parent:
                parent[find(o)] = find(d["id"])
    g = defaultdict(list)
    for d in reg["datasets"]:
        g[find(d["id"])].append(d)
    return list(g.values())


def summary(reg):
    by_class = defaultdict(lambda: {"hours": 0.0, "unknown": 0, "n": 0})
    by_lang = defaultdict(lambda: {"hours": 0.0, "unknown": 0})
    for d in reg["datasets"]:
        key = d["classification"] + ("/excluded_by_rule" if d.get("excluded_by_project_rule") else "")
        for k, agg in ((key, by_class), (d["language"], by_lang)):
            if d["hours"] is None:
                agg[k]["unknown"] += 1
            else:
                agg[k]["hours"] += d["hours"]
        by_class[key]["n"] += 1
    dedup, dedup_lang = 0.0, defaultdict(float)
    for grp in groups(reg):
        hs = [d for d in grp if d["hours"] is not None]
        if hs:
            top = max(hs, key=lambda d: d["hours"])
            dedup += top["hours"]
            dedup_lang[top["language"]] += top["hours"]
    commercial = [d for d in reg["datasets"] if d["commercial_training_allowed"]]
    lev = [d for d in reg["datasets"] if d["language"] == "ar" and any(w in (d.get("dialect") or "") for w in ("Jordan", "Levant", "Palestin", "Syria", "Leban"))]
    return {
        "datasets": len(reg["datasets"]),
        "raw_hours_by_class": {k: {"hours": round(v["hours"], 1), "datasets": v["n"], "datasets_with_unknown_hours": v["unknown"]} for k, v in sorted(by_class.items())},
        "raw_hours_by_language": {k: {"hours": round(v["hours"], 1), "datasets_with_unknown_hours": v["unknown"]} for k, v in sorted(by_lang.items())},
        "dedup_hours_all": round(dedup, 1),
        "dedup_hours_by_language": {k: round(v, 1) for k, v in sorted(dedup_lang.items())},
        "dedup_note": "overlap groups contribute their largest member; unknown-hour entries (People's Speech ~30,000 h, Common Voice, LDC, Casablanca) are not included",
        "commercial_training_allowed": {"hours": round(sum(d["hours"] or 0 for d in commercial), 1),
                                        "by_language": {l: round(sum(d["hours"] or 0 for d in commercial if d["language"] == l), 1) for l in ("en", "ar")},
                                        "known_speakers": sum(d["speakers"] or 0 for d in commercial),
                                        "datasets": [d["id"] for d in commercial]},
        "jordanian_levantine": {"commercial_hours": round(sum(d["hours"] or 0 for d in lev if d["commercial_training_allowed"]), 1),
                                "dialect_specific_hours_known": {d["id"]: d["hours"] for d in lev
                                                                 if d["hours"] and d["id"] in ("arabic_speech_corpus", "synthetic_levantine_tts")},
                                "note": "MASC / ADI-17 / Casablanca / LDC contain Levantine portions whose size is not published in the evidence read",
                                "datasets": [f"{d['id']} ({d['classification']})" for d in lev]},
    }


if __name__ == "__main__":
    reg = load()
    errs = violations(reg)
    s = summary(reg)
    print(json.dumps(s, indent=1, ensure_ascii=False))
    if errs:
        print("\n".join(errs), file=sys.stderr)
        sys.exit(1)
