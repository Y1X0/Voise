# LLVC (pretrained) on Arabic: exploratory test

**Decision: CONTINUE, as an architecture to adapt to Arabic. Not usable as-is.**

**The results are exploratory:** 12 speakers, 60 clips. They are not evidence of privacy and do
not pass the project's acceptance protocol, which needs at least 40 speakers and the A1–A6
attacks. `data/acceptance_criteria.json` is unchanged, and Android/DSP was not modified.

- Raw results: `llvc_svq_report.json`
- Script: `llvc_svq_test.py`
- Run: Kaggle kernel (private), T4 + 1 CPU thread, 2026-10-10

## What was run (verified)

| Item | Detail |
|---|---|
| Model | LLVC `G_500000.pth` (39.5 MB): KoeAI/LLVC commit 1627c5d, MIT code and weights, pretrained only, no training/fine-tuning |
| Data | google/svq (CC BY 4.0; SVQ, Heigold et al., MSEB) |
| Selection | `clean` condition, 4 Arabic locales × 3 speakers × 5 clips = 12 speakers (6 F / 6 M), 60 clips, 284 s; speakers appearing in more than one locale excluded; seed 20261010 |
| Bytes transferred | 1.12 GB of compressed audio was read inside Kaggle, because SVQ row groups are large; only the 60 clips were kept, nothing else was stored |
| Speaker separation | no training/tuning, so no split; every trial pairs *different* utterances |
| Independent ASV | SpeechBrain ECAPA-VoxCeleb (Apache-2.0) and WavLM-base-plus-sv (CC BY-SA 3.0); neither is part of LLVC |
| Intelligibility | whisper-small, Arabic, CER against the SVQ reference text |

## Measured results

### Speed

RTF = processing time / audio duration.

| Setting | Chunk | Algorithmic latency | Chunk time p50 / p95 / max | RTF | Chunks over deadline | Max lag |
|---|---|---|---|---|---|---|
| CPU, 1 thread (Xeon 2.0 GHz), chunk ×1 | 13 ms | 15 ms | 17.8 / 19.1 / 33.5 ms | **1.38** | 100 % | 3.37 s (grows) |
| CPU, 1 thread, chunk ×2 | 26 ms | 28 ms | 18.9 / 20.4 / 34.6 ms | **0.74** | 0.5 % | 41 ms (no build-up) |
| T4 GPU, chunk ×1 | 13 ms | 15 ms | 8.1 / 9.0 / 17.3 ms | 0.64 | 0.2 % | 19 ms |
| T4 GPU, chunk ×2 | 26 ms | 28 ms | 8.6 / 9.4 / 17.8 ms | 0.33 | 0 % | 18 ms |

Streaming output vs full-utterance output: max abs difference 0.008.

### Intelligibility (whisper-small CER)

| | Original | LLVC |
|---|---|---|
| Median | 0.200 | **0.445** |
| Mean (inflated by Whisper hallucinations on short clips) | 0.561 | 1.231 |
| Gulf (mean) | 0.172 | 0.358 |
| Levantine (mean) | 0.132 | 0.363 |
| Egyptian (mean) | 0.165 | 1.753 |

### Speaker change

EER with a 95 % speaker-bootstrap CI (1000 resamples). 0.5 means the speakers can't be told
apart.

| Trial | ECAPA | WavLM-SV |
|---|---|---|
| Original vs original (sanity) | 0.034 [0.004, 0.100] | 0.133 [0.067, 0.210] |
| Ignorant: original enrolment vs LLVC trial | **0.417** [0.256, 0.498] | **0.500** [0.347, 0.609] |
| Lazy-informed: LLVC vs LLVC | **0.233** [0.134, 0.314] | **0.350** [0.238, 0.444] |
| Mean cosine, same utterance original vs LLVC | 0.258 | 0.689 |
| Mean cosine, LLVC vs LLVC, different speakers | 0.525 | 0.917 |

## Interpretation (estimates, not measurements)

- **Speed:** on 1 server CPU thread it keeps up only at chunk ×2 (28 ms latency). A phone big core
  is not measured: it could be faster or slower. The ops are standard
  (Conv1d / ConvTranspose1d / LayerNorm / causal Transformer decoder), so export is plausible.
- **Speaker change:**
  - The voice moves far from the original (ignorant EER ≈ 0.42–0.50).
  - Converted speakers remain partly linkable to each other (lazy-informed EER 0.23–0.35). ECAPA's
    point value is below the project's informed-attacker threshold of 0.25, and both CIs are wide.
- **Intelligibility:** the pretrained (English-trained) model roughly doubles CER on Arabic. On this
  evidence, Arabic use as-is is not acceptable.

## Why CONTINUE

The architecture already meets the hard constraints: causal, small, standard ops, and real-time
on 1 CPU thread at 28 ms. It also changes speaker identity strongly. Its weakness, Arabic
intelligibility, is what Arabic adaptation targets.

Adaptation means distilling LLVC on Arabic input paired with converted targets toward a
synthetic pseudo-speaker. It needs training, and that is a separate decision.

**Kill criteria for the adaptation step:**
- Arabic CER must come back near the original;
- lazy-informed EER must stay at least at today's level;
- it must stay real-time on 1 thread.

If not, REJECT LLVC and return to StreamAnon-S.
