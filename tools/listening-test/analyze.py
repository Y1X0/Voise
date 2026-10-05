#!/usr/bin/env python3
"""Aggregate blind listening-test ratings.

usage: analyze.py KEY_experimenter_only.json ratings_*.csv

Joins each listener's CSV (from index.html) with the session key, then prints
per-condition mean, standard deviation, n and a 95 % t-interval for each
criterion, plus how often each condition was guessed to be the original.

With few listeners the intervals are wide; results are labelled
"exploratory" when n < 10 and must not be presented as strong evidence.
"""
import csv
import json
import math
import sys
from collections import defaultdict

CRITERIA = ["intelligibility", "naturalness", "similarity", "artifacts", "anonymity"]
# Two-sided 95 % Student-t critical values for df = 1..30.
T95 = [12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13,
       2.12, 2.11, 2.10, 2.09, 2.09, 2.08, 2.07, 2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.05, 2.04]


def summary(values):
    n = len(values)
    if n == 0:
        return "n=0"
    mean = sum(values) / n
    if n == 1:
        return f"{mean:.2f} (n=1)"
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    t = T95[min(n - 1, 30) - 1]
    half = t * sd / math.sqrt(n)
    return f"{mean:.2f} ± {half:.2f} (sd {sd:.2f}, n={n})"


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    keys = {}
    for k in [a for a in argv[1:] if a.endswith(".json")]:
        data = json.load(open(k, encoding="utf-8"))
        keys[data["session"]] = data["mapping"]
    if len(keys) == 1:
        only = next(iter(keys.values()))
    scores = defaultdict(lambda: defaultdict(list))
    guessed = defaultdict(int)
    listeners = set()
    for path in [a for a in argv[1:] if a.endswith(".csv")]:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            mapping = keys.get(row["session"].replace("listening-test-", ""), None) or (only if len(keys) == 1 else None)
            if mapping is None:
                print(f"warning: no key for session '{row['session']}' in {path}", file=sys.stderr)
                continue
            cond = mapping[row["file"].replace(".wav", "")]
            listeners.add(row["participant"])
            for c in CRITERIA:
                scores[cond][c].append(int(row[c]))
            guessed[cond] += int(row["guessed_original"])
    n = len(listeners)
    label = "EXPLORATORY (n < 10 listeners) - do not generalise" if n < 10 else "descriptive"
    print(f"Listeners: {n}  [{label}]\n")
    print("| condition | " + " | ".join(CRITERIA) + " | guessed as original |")
    print("|---|" + "---|" * (len(CRITERIA) + 1))
    for cond in sorted(scores):
        print(f"| {cond} | " + " | ".join(summary(scores[cond][c]) for c in CRITERIA) + f" | {guessed[cond]}/{n} |")
    print("\nScales: 1-5, higher = better for intelligibility, naturalness, artifacts(=fewer), anonymity;")
    print("similarity: higher = more similar to the reference speaker (lower means more change).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
