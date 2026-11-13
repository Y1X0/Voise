"""Step 2 for a multi-speaker recording, after segment.py: group voices, drop non-Jordanian dialects, match each
voice to a person already in the learning pool (same player in an earlier episode) and cut clean clips per voice.

  python prepare_episode.py EPISODE_DIR TAG [--pool runs/learn/pool]

Writes EPISODE_DIR/src_<person>/ (one folder per kept voice) and EPISODE_DIR/summary.json. A voice whose centre is
close to a pool person's (ECAPA cosine >= MATCH) is filed under that person, so the same player across episodes
stays one person (one revocation removes all of them); other voices get <TAG>_p<group>.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "scripts"))
import voice_clone_tts as V  # noqa: E402

SR = 16000
# forms that are clearly not Jordanian (North African, Iraqi, Egyptian); Levantine forms shared with Jordanian
# (ليه، برضه، عارف) are not on the list
FOREIGN = (r"(^|\s)(شنو|علاش|باهي|ماكو|بركي|هيدي|وش|كليش|مكشفت|ڨ\w*|سامسوش|نقول|كده|إزاي|بتاع)"
           r"($|[\s،,.؟?!])")
CLUSTER_DIST, FOREIGN_SHARE, MIN_VOICE_S, MATCH, PURITY = 0.6, 0.06, 45, 0.6, 0.55


def pool_centres(pool, enc, cache):
    """Mean ECAPA embedding per pool person (cached per clip, so each pool clip is embedded once)."""
    try:
        emb = json.load(open(cache))
    except FileNotFoundError:
        emb = {}
    centres = {}
    for d in sorted(glob.glob(os.path.join(pool, "*"))):
        vs = []
        for w in sorted(glob.glob(os.path.join(d, "*.wav")))[:40]:
            k = os.path.relpath(w, pool)
            if k not in emb:
                emb[k] = np.asarray(V.speaker_embedding(enc, sf.read(w, dtype="float32")[0])).tolist()
            vs.append(np.asarray(emb[k]))
        if vs:
            c = np.mean(vs, 0)
            centres[os.path.basename(d)] = c / np.linalg.norm(c)
    json.dump(emb, open(cache, "w"))
    return centres


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("tag")
    ap.add_argument("--pool", default=os.path.join(V.RUNS, "learn", "pool"))
    a = ap.parse_args(argv)
    S = json.load(open(os.path.join(a.dir, "segments.json"), encoding="utf-8"))
    E = np.load(os.path.join(a.dir, "emb.npy")); E = E / np.linalg.norm(E, axis=1, keepdims=True)
    lab = fcluster(linkage(E, "average", metric="cosine"), CLUSTER_DIST, "distance")
    wav = glob.glob(os.path.join(a.dir, "*.wav"))[0]
    y = sf.read(wav, dtype="float32")[0]
    enc = V.load_speaker_encoder("models/ecapa-voxceleb")
    centres = pool_centres(a.pool, enc, os.path.join(V.RUNS, "learn", "pool_emb.json"))
    groups = collections.defaultdict(list)
    for i, l in enumerate(lab):
        groups[int(l)].append(i)
    summary, used = {"kept": {}, "dropped_dialect": {}, "too_short": 0}, set()
    for g, idx in sorted(groups.items(), key=lambda x: -sum(S[i]["b"] - S[i]["a"] for i in x[1])):
        dur = sum(S[i]["b"] - S[i]["a"] for i in idx)
        if dur < MIN_VOICE_S:
            summary["too_short"] += 1
            continue
        foreign = [S[i]["text"] for i in idx if re.search(FOREIGN, S[i]["text"])]
        if len(foreign) / len(idx) > FOREIGN_SHARE:
            summary["dropped_dialect"][g] = dict(minutes=round(dur / 60, 1), examples=foreign[:3])
            continue
        c = E[idx].mean(0); c /= np.linalg.norm(c)
        match = max(centres.items(), key=lambda kv: float(kv[1] @ c), default=(None, None))
        sim = float(match[1] @ c) if match[0] else 0.0
        name = match[0] if match[0] and sim >= MATCH and match[0] not in used else f"{a.tag}_p{g}"
        used.add(name)
        out = os.path.join(a.dir, f"src_{name}"); os.makedirs(out, exist_ok=True); n = sec = 0
        for i in idx:
            s = S[i]
            if s["b"] - s["a"] < 2 or float(E[i] @ c) < PURITY or re.search(FOREIGN, s["text"]):
                continue
            sf.write(os.path.join(out, f"{a.tag}_{i:04d}.wav"), y[int(s["a"] * SR):int(s["b"] * SR)], SR, subtype="PCM_16")
            n += 1; sec += s["b"] - s["a"]
        summary["kept"][name] = dict(group=g, clips=n, minutes=round(sec / 60, 1),
                                     matched_pool_person=match[0] if name == match[0] else None,
                                     similarity_to_closest=round(sim, 3), closest=match[0])
    json.dump(summary, open(os.path.join(a.dir, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
