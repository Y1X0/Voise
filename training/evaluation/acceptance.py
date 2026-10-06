#!/usr/bin/env python3
"""Final acceptance evaluation against the LOCKED criteria (data/acceptance_criteria.json).

Input: a results JSON {metric: {"value": x, "source": "MEASURED"|..., optional "ci_lower",
"n_speakers", "n_raters", "measured_on_device", "device", "evidence"}}. Output per metric:
  PASS          measured value meets the threshold (and its CI lower bound / sample-size rule)
  FAIL          measured value misses it. Thresholds are NEVER redefined.
  NOT_MEASURED  no value, or source != MEASURED
  INVALID       measured under the wrong conditions (e.g. Top-1 with < 40 speakers, RTF not on a
                real Android device, MOS with too few raters)
Overall verdict: PASS only if every criterion is PASS; FAIL if any FAIL; else INCOMPLETE.

  python3 training/evaluation/acceptance.py --results runs/s/final_results.json --out runs/s/acceptance
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CRITERIA = os.path.join(ROOT, "data", "acceptance_criteria.json")
LOCKED_SHA256 = "58a893c21543d2075b4d962014e01cac093e9cf625af2a5b1b8a2fdd13af1aaf"


class CriteriaTampered(RuntimeError):
    pass


def load_criteria(path=CRITERIA, pinned=LOCKED_SHA256):
    with open(path, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != pinned:
        raise CriteriaTampered(f"{path} changed (sha256 != pinned); criteria are locked")
    return json.loads(raw)["criteria"]


def _meets(direction, v, thr):
    if direction == "true":
        return v is True
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return {"min": v >= thr, "max": v <= thr, "max_exclusive": v < thr}[direction]


def evaluate(results, criteria=None):
    criteria = criteria or load_criteria()
    out = {}
    for name, c in criteria.items():
        r = results.get(name)
        row = {"criterion": c, "value": None if r is None else r.get("value"), "status": None, "target_met": None}
        if r is None or r.get("value") is None or r.get("source") != "MEASURED":
            row["status"] = "NOT_MEASURED"
        elif c.get("requires_device") and not (r.get("measured_on_device") is True and r.get("device")):
            row["status"] = "INVALID"
            row["reason"] = "must be measured on a real Android device (big core)"
        elif "min_speakers" in c and int(r.get("n_speakers", 0)) < c["min_speakers"]:
            row["status"] = "INVALID"
            row["reason"] = f"needs >= {c['min_speakers']} speakers"
        elif "min_raters" in c and int(r.get("n_raters", 0)) < c["min_raters"]:
            row["status"] = "INVALID"
            row["reason"] = f"needs >= {c['min_raters']} raters"
        else:
            v = r["value"]
            ok = _meets(c["direction"], v, c.get("threshold"))
            if ok and "ci_lower_min" in c:
                lb = r.get("ci_lower")
                if lb is None:
                    ok, row["reason"] = False, "95 % CI lower bound required"
                elif lb < c["ci_lower_min"]:
                    ok, row["reason"] = False, f"CI lower bound {lb} < {c['ci_lower_min']}"
            row["status"] = "PASS" if ok else "FAIL"
            if "target" in c and c["direction"] in ("min", "max"):
                row["target_met"] = _meets(c["direction"], v, c["target"])
        out[name] = row
    st = [r["status"] for r in out.values()]
    verdict = "PASS" if all(s == "PASS" for s in st) else ("FAIL" if "FAIL" in st else "INCOMPLETE")
    return {"verdict": verdict, "criteria_sha256": LOCKED_SHA256, "results": out}


def to_markdown(ev):
    lines = [f"# Acceptance evaluation: **{ev['verdict']}**", "", f"criteria sha256 `{ev['criteria_sha256']}` (locked)", "",
             "| Criterion | Threshold | Value | Status | Target met |", "|---|---|---|---|---|"]
    for n, r in ev["results"].items():
        c = r["criterion"]
        thr = {"min": "≥ ", "max": "≤ ", "max_exclusive": "< ", "true": ""}[c["direction"]] + (str(c.get("threshold")) if c["direction"] != "true" else "true")
        lines.append(f"| {n} | {thr} | {r['value']} | {r['status']}{' (' + r['reason'] + ')' if r.get('reason') else ''} | {r['target_met']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.results) as f:
        ev = evaluate(json.load(f))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "acceptance.json"), "w") as f:
        json.dump(ev, f, indent=1)
    with open(os.path.join(a.out, "acceptance.md"), "w") as f:
        f.write(to_markdown(ev))
    print(ev["verdict"])
    sys.exit(0 if ev["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
