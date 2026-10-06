#!/usr/bin/env python3
"""Feed audio rendered offline by another tool (e.g. vpc_b3_render.py) into
scripts/neural_anonymization_eval.py: copies <root>/<session>/<utt>.wav to {out}.

  "name=cmd:python3 scripts/model_adapters/prerendered.py --root DIR --in {in} --out {out} --seed {seed}"
"""
import argparse
import os
import shutil
import sys

SESSION_OF_SEED = {101: "A", 202: "B", 303: "C"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
src = os.path.join(a.root, SESSION_OF_SEED[a.seed], os.path.basename(a.inp))
if not os.path.exists(src):
    sys.exit(f"missing pre-rendered file {src}")
shutil.copyfile(src, a.out)
