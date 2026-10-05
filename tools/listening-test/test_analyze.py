#!/usr/bin/env python3
"""Self-test for analyze.py using dummy ratings (not real listener data)."""
import json
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
with tempfile.TemporaryDirectory() as d:
    key = os.path.join(d, "KEY_experimenter_only.json")
    json.dump({"session": "abc", "sampleRate": 16000, "direction": 1,
               "mapping": {"sample_1": "C_balanced", "sample_2": "A_original",
                           "sample_3": "D_strong", "sample_4": "B_natural"}}, open(key, "w"))
    csvs = []
    for p, vals in [("P1", [4, 5, 3, 4]), ("P2", [3, 5, 2, 4])]:
        path = os.path.join(d, f"ratings_{p}.csv")
        with open(path, "w") as f:
            f.write("participant,knows_speaker,session,file,presented_as,intelligibility,naturalness,"
                    "similarity,artifacts,anonymity,guessed_original\n")
            for i, v in enumerate(vals):
                f.write(f"{p},no,listening-test-abc,sample_{i+1}.wav,clip{i+1},{v},{v},{6-v},{v},{6-v},"
                        f"{1 if i == 1 else 0}\n")
        csvs.append(path)
    out = subprocess.run([sys.executable, os.path.join(here, "analyze.py"), key, *csvs],
                         capture_output=True, text=True, check=True).stdout
    assert "Listeners: 2" in out and "EXPLORATORY" in out, out
    assert "| A_original | 5.00" in out and "2/2" in out, out
    assert "| D_strong | 2.50" in out, out
print("analyze.py self-test passed")
