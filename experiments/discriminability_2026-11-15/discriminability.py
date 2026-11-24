"""Roadmap step 1: does shared training make different people's clones sound alike?

For every voice in the pool, the same sentences are spoken by the base model and by the current learned version
(same reference, seed 0, one take, no polish), and each clone is placed among the REAL voices:

  identification   is the real voice closest to the clone the right person? (top-1 over all voices)
  own              the clone's similarity to its own person's real centre
  others           its similarity to the nearest OTHER person's real centre
  margin           own - others: how clearly the clone is that person and nobody else
  spread           mean similarity between different people's clones (higher = the clones sound more alike), next
                   to the same number for the real people's centres

A generation that raises "own" but shrinks "margin" or raises "spread" is blurring identities.

Usage: python discriminability.py OUT_DIR [adapter ...]   ("none" = the base model)
"""
import glob, itertools, json, os, sys
import librosa, numpy as np, soundfile as sf, torch

sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402

SR, POOL = 16000, "runs/learn/pool"
TEXTS = ["والله يا جماعة ما توقعت إنه هيك رح تصير، بس خلص، المهم إنه كلنا مع بعض.",
         "شوف، أنا رأيي إنه نستنى شوي، لأنه يعني لسا بكير نقرر."]


def main(out, adapters):
    os.makedirs(out, exist_ok=True)
    enc = V.load_speaker_encoder("models/ecapa-voxceleb")
    people = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    centres = {}
    for p in people:
        es = [V.speaker_embedding(enc, sf.read(w, dtype="float32")[0]) for w in sorted(glob.glob(f"{POOL}/{p}/*.wav"))[:30]]
        c = sum(es) / len(es); centres[p] = c / c.norm()
    real_spread = float(np.mean([float(centres[a] @ centres[b]) for a, b in itertools.combinations(people, 2)]))
    res = {"people": people, "real_spread": round(real_spread, 3), "versions": {}}
    print("real spread", round(real_spread, 3), flush=True)
    for ad in adapters:
        eng = V.Engine("models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2",
                       adapter=None if ad == "none" else ad, sv_dir="models/ecapa-voxceleb")
        eng.POLISH_BELOW = 0.0                                   # measure the model itself, no voice conversion
        name = "base" if ad == "none" else os.path.basename(ad)
        clone_emb, rows = {}, []
        for p in people:
            v = eng.voice(os.path.join(POOL, p))
            es = []
            for t in TEXTS:
                best, _ = eng.takes(v, t, candidates=1)
                y16 = librosa.resample(np.asarray(best["y"], np.float32), orig_sr=eng.sr, target_sr=SR)
                es.append(V.speaker_embedding(enc, y16))
            e = sum(es) / len(es); e = e / e.norm(); clone_emb[p] = e
            sims = {q: float(e @ centres[q]) for q in people}
            other = max((q for q in people if q != p), key=lambda q: sims[q])
            rows.append(dict(person=p, own=round(sims[p], 3), nearest_other=other, others=round(sims[other], 3),
                             margin=round(sims[p] - sims[other], 3), identified=max(sims, key=sims.get) == p))
            print("V", name, rows[-1], flush=True)
        spread = float(np.mean([float(clone_emb[a] @ clone_emb[b]) for a, b in itertools.combinations(people, 2)]))
        res["versions"][name] = dict(rows=rows, identified=sum(r["identified"] for r in rows),
                                     own=round(float(np.mean([r["own"] for r in rows])), 3),
                                     margin=round(float(np.mean([r["margin"] for r in rows])), 3),
                                     clone_spread=round(spread, 3))
        print("S", name, {k: v for k, v in res["versions"][name].items() if k != "rows"}, flush=True)
        json.dump(res, open(os.path.join(out, "discriminability.json"), "w"), ensure_ascii=False, indent=1)
        del eng
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
