"""Does restoration keep each speaker's identity? ECAPA speaker embeddings per condition.

For each condition (raw / sepformer / sidon), on speech-only (VAD-compacted) clips of >= 2 s:
  same     mean cosine between clips of the SAME speaker   (higher = the person sounds consistent)
  diff     mean cosine between clips of DIFFERENT speakers (lower = people stay distinguishable)
  gap      same - diff
  eer      equal error rate of same/different decisions over all clip pairs (lower = identity better kept)
  keep     mean cosine between a clip and ITS OWN raw version (how much restoration moved the voice)

Usage: python identity.py RESTORED_DIR ECAPA_DIR OUT_JSON
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A  # noqa: E402

SPEAKERS = ("khaled", "muhannad", "mohammad", "majd", "shahd")


def eer(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels, bool)
    best = 1.0
    for t in np.unique(s):
        far, frr = np.mean(s[~l] >= t), np.mean(s[l] < t)
        best = min(best, max(far, frr))
    return float(best)


def main(root, ecapa_dir, out_json):
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir + "_sv", run_opts={"device": "cpu"})
    vad = A.load_vad()

    def emb(p, keep_regions=None):
        y, _ = sf.read(p, dtype="float32")
        regs = keep_regions if keep_regions is not None else A.speech_regions(y, vad)
        t, _ = A.compact(y, regs, max_pause_s=0.3)
        with torch.no_grad():
            e = ec.encode_batch(torch.as_tensor(t)[None]).squeeze()
        return torch.nn.functional.normalize(e, dim=-1).numpy(), len(t), regs

    # speech regions are found once on the RAW clip and reused, so every condition scores the same audio span
    E, regions = {}, {}
    for cond in ("raw", "sepformer", "sidon"):
        E[cond] = {}
        for spk in SPEAKERS:
            for p in sorted(glob.glob(os.path.join(root, cond, spk, "*.wav"))):
                key = (spk, os.path.basename(p))
                e, n, regs = emb(p, regions.get(key))
                if cond == "raw":
                    if n < 2 * 16000:
                        continue
                    regions[key] = regs
                elif key not in regions:
                    continue
                E[cond][key] = e
    res = {}
    for cond, d in E.items():
        keys = sorted(d)
        pairs = list(itertools.combinations(keys, 2))
        sc = [float(d[a] @ d[b]) for a, b in pairs]
        lab = [a[0] == b[0] for a, b in pairs]
        same = [s for s, l in zip(sc, lab) if l]
        diff = [s for s, l in zip(sc, lab) if not l]
        per = {spk: round(float(np.mean([s for (a, b), s in zip(pairs, sc) if a[0] == b[0] == spk])), 3) for spk in SPEAKERS}
        res[cond] = dict(clips=len(keys), same=round(float(np.mean(same)), 3), diff=round(float(np.mean(diff)), 3),
                         gap=round(float(np.mean(same) - np.mean(diff)), 3), eer=round(eer(sc, lab), 3),
                         keep=round(float(np.mean([float(d[k] @ E["raw"][k]) for k in keys])), 3), per_speaker_same=per)
        print(cond, res[cond], flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
