"""Build a large kNN-VC matching set from ClArTTS train shards (CC BY 4.0; one male LibriVox reader).
Streams each parquet shard row by row (the audio column is float64 lists, too big to load whole), resamples
to 16 kHz, and caches per shard: WavLM-Large layer-6 features (float16) + 16 kHz int16 audio + offsets.

  VOISE_DL=... python3 experiments/knnvc_matching_2026-10-24/build_matching.py OUT_DIR SHARD [SHARD ...]
"""
import os, sys, time
import numpy as np, torch, torchaudio
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "experiments/knnvc_2026-10-24"))
from knnvc_lib import load_wavlm, wavlm_feats
torch.set_grad_enabled(False)
SR = 16000
OUT, shards = sys.argv[1], sys.argv[2:]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
wl = load_wavlm()
for path in shards:
    name = os.path.basename(path).split(".")[0]
    dst = os.path.join(OUT, name + ".pt")
    if os.path.exists(dst):
        print("skip (cached)", dst, flush=True); continue
    pf = pq.ParquetFile(path)
    feats, audio, files, f_off, a_off = [], [], [], [0], [0]
    for batch in pf.iter_batches(batch_size=4, columns=["file", "audio", "sampling_rate"]):
        col = batch.column("audio")
        offs = col.offsets.to_numpy(); vals = col.values.to_numpy(zero_copy_only=False)
        for i in range(batch.num_rows):
            a = vals[offs[i]:offs[i + 1]].astype(np.float32)
            sr = int(batch.column("sampling_rate")[i].as_py())
            y = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
            f = wavlm_feats(wl, y).to(torch.float16)
            feats.append(f); audio.append(np.clip(y * 32767, -32768, 32767).astype(np.int16))
            files.append(batch.column("file")[i].as_py())
            f_off.append(f_off[-1] + len(f)); a_off.append(a_off[-1] + len(y))
    torch.save(dict(feats=torch.cat(feats), audio=torch.from_numpy(np.concatenate(audio)), files=files,
                    feat_offsets=torch.tensor(f_off), audio_offsets=torch.tensor(a_off), source=os.path.basename(path)), dst)
    print(f"[{time.time() - t0:6.0f}s] {name}: {len(files)} utterances, {a_off[-1] / SR / 60:.1f} min, {f_off[-1]} frames", flush=True)
