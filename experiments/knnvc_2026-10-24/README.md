# kNN-VC on the owner's recording (2026-10-24): exploratory result

Inference only. Nothing here was trained. The audio and reports stay under the git-ignored `runs/`.

## What was run

- **Method:** kNN-VC (Baas et al., 2023), re-implemented locally in `knnvc_lib.py`.
  - Each WavLM-Large layer-6 frame of the source is replaced by the mean of its k nearest frames (cosine) from a target speaker's matching set.
  - The result is vocoded with the released prematched HiFi-GAN.
- **Released weights:** github.com/bshall/knn-vc v0.1. Both files are loaded with `weights_only=True`, and no code from that repository is used.
  - `WavLM-Large.pt` sha256 `6fb4b3c3…100f`
  - `prematch_g_02500000.pt` sha256 `f924c763…2dd5`
- **Source:** `runs/inputs/my_voice.wav`, one speaker (the owner), 52.5 s after trimming digital silence.
- **Arabic target:** ClArTTS test split (MBZUAI/ClArTTS @2f6e820a). It is CC BY 4.0, one male LibriVox reader, from Kulkarni et al., Interspeech 2023.
  - Matching pool: first 10.1 min.
  - Speaker reference: 60 s, held apart from the pool.
- **Measures:**
  - Whisper-small (@973afd24) transcripts of each conversion, compared with Whisper's transcript of the original. This gives relative CER/WER, since no human transcript exists.
  - WavLM-base-plus-SV (@feb593a6) cosine similarity.

## Results (`eval_arabic.py`)

### Same 10.5 s held-out part, all systems

| System | CER | WER | sim to owner | sim to target reader |
|---|---|---|---|---|
| original | 0 | 0 | 0.985 | 0.891 |
| unit vocoder, after normalisation fix (`runs/own_2026-10-23_fix1`) | 0.549 | 0.909 | 0.877 | 0.781 |
| kNN-VC to WORLD pseudo-speaker | 0.088 | 0.273 | 0.985 | 0.884 |
| kNN-VC to 3 LibriSpeech speakers (45 s, English) | 0.363 | 0.636 | 0.861 | 0.863 |
| **kNN-VC to ClArTTS reader, 10 min, k=4** | **0.029** | **0.182** | 0.881 | **0.936** |

### Full 52.5 s recording, ClArTTS target

| Matching pool, k | CER | WER | sim to owner | sim to target reader |
|---|---|---|---|---|
| 1 min, k=4 | 0.158 | 0.446 | 0.893 | 0.974 |
| 3 min, k=4 | 0.115 | 0.416 | 0.894 | 0.982 |
| 10 min, k=2 | **0.100** | **0.347** | 0.889 | 0.957 |
| 10 min, k=4 | 0.139 | 0.366 | 0.891 | 0.953 |
| 10 min, k=8 | 0.269 | 0.436 | 0.890 | 0.948 |

### Owner listening

The owner judged the ClArTTS conversions intelligible and reported that they do not sound like themself.

## Reading these numbers

**Clarity.** The Arabic target with a few minutes of matching audio is the clearest system tried. More matching audio helps, and k=2 is the clearest setting.

**Speaker identity: NOT established.**
- The verifier separates these voices poorly: the owner and the target reader, two different people, already score 0.861.
- Converted audio moves toward the target (0.95) and away from the owner (0.99 → 0.89), but it stays above that 0.861 level.
- The test uses one speaker, one 55 s recording, no attacker model and no EER.
- The owner's own listening is evidence, not proof, because people perceive their own recorded voice differently.

**What would establish identity hiding:**
- a blind recognition test by listeners who know the owner;
- an EER over many Arabic speakers, using a stronger verifier, under lazy-informed and semi-informed attackers.

## Files

| File | Purpose |
|---|---|
| `knnvc_lib.py` | WavLM key mapping, generator and kNN step |
| `knnvc_run.py` | Matching set = WORLD renders of the training chunks |
| `knnvc_other.py`, `knnvc_k.py` | LibriSpeech targets and k sweep |
| `spk_proxy.py` | Early unvalidated WavLM-mean similarity proxy, superseded by WavLM-SV |
| `eval_arabic.py` | ClArTTS target plus Whisper and WavLM-SV measures |

The scripts expect the downloaded weights and data under `$VOISE_DL` (`knnvc_weights/`, `hf/`).
