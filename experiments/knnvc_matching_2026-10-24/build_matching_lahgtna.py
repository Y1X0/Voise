"""Matching set from Lahgtna Levantine TTS (CC BY 4.0; SYNTHETIC voices, each cloned by the dataset authors from
5-15 s of a real Levantine speaker whose consent is not documented). Only `pure_levantine` utterances, up to
--minutes per speaker; the speakers are later POOLED so the converted voice is a blend, not one cloned person.
Research test of the dialect effect only.

  VOISE_DL=... python3 build_matching_lahgtna.py OUT_DIR --minutes 20 SHARD [SHARD ...]
"""
import argparse, io, os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "experiments/knnvc_2026-10-24"))
from knnvc_lib import load_wavlm, wavlm_feats
torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("shards", nargs="+"); ap.add_argument("--minutes", type=float, default=20)
a = ap.parse_args()
SR = 16000
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
wl = load_wavlm()
for path in a.shards:
    pf = pq.ParquetFile(path)
    feats, spk, secs, texts = [], None, 0.0, []
    for batch in pf.iter_batches(batch_size=8, columns=["audio", "speaker_id", "sentence_type", "text"]):
        d = batch.to_pydict()
        for au, s, st, tx in zip(d["audio"], d["speaker_id"], d["sentence_type"], d["text"]):
            if st != "pure_levantine" or secs >= a.minutes * 60:
                continue
            y, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
            y = y if y.ndim == 1 else y.mean(1)
            y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR).numpy()
            feats.append(wavlm_feats(wl, y).to(torch.float16)); spk = s; secs += len(y) / SR; texts.append(tx)
        if secs >= a.minutes * 60:
            break
    dst = os.path.join(a.out, f"{spk}.pt")
    torch.save(dict(feats=torch.cat(feats), speaker=spk, seconds=secs, n=len(feats), source=os.path.basename(path)), dst)
    print(f"[{time.time() - t0:6.0f}s] {spk}: {len(feats)} utterances, {secs / 60:.1f} min", flush=True)
