# Training compute estimate (StreamAnon-S)

The inputs are **measured** where this environment allows it, and **derived** where it
needs a GPU. Each number is labelled.

* **Measurement:** `training/scripts/measure_training_cost.py`, run on CPU at the real
  batch size and with the real model and discriminator sizes.
* **Raw output:** `docs/results/training_cost_measured.json`.
* **Not available here:** no GPU, so GPU throughput is derived from the measured FLOPs
  under stated assumptions. It must be replaced by the first 1 000 real GPU steps.
* **Prices:** none were verified, so every cost cell is **PRICE_NOT_VERIFIED** and the
  formula is given instead.

This document **supersedes** the GPU-hour figures in `STREAMING_NEURAL_TRAINING_PLAN.md` §7.

## 1. Measured per-step cost (batch 16 × 2.0 s, 16 kHz)

| Item | Measured | How |
|---|---|---|
| Stage 1 step (encoder + VQ + unit head, fwd + bwd) | **0.042 TFLOP**; 0.29 GB saved activations (fp32) | FlopCounterMode / saved-tensor hooks |
| Stage 2 step (generator + full MPD/MRSD discriminators, G and D updates) | **4.10 TFLOP**; **9.95 GB** saved activations (fp32) | same |
| Stage 3 step, generator + discriminators + adversaries (smoke placeholders for the frozen models) | 5.58 TFLOP; 14.2 GB (fp32) | same |
| Content teacher at real size (HuBERT-base, 94.4 M params), forward on one batch | 0.44 TFLOP | random weights; FLOPs do not depend on them |
| Speaker encoder at real size (ECAPA-TDNN C=1024, 20.8 M params), forward on one batch | 0.12 TFLOP | same |
| **Stage 3 step at real size** (teacher on x and ŷ incl. backward through ŷ ≈ 4× fwd; 2 speaker encoders likewise) | **≈ 8.3 TFLOP** (derived from the rows above) | — |
| Trainable params | generator + conditioning encoder 7.03 M; discriminators 15.16 M | counted |
| Params + grads + Adam moments (fp32) | 0.34 GB | 16 B per trainable param |
| Frozen models (fp16) | 0.26 GB | — |

* **Discriminators dominate stage 2/3 cost.** The generator itself is ~7 M params.
  Smaller MPD channels are a cost lever (an ablation, not a default).

## 2. Data (commercial path)

| Scenario | Corpora | Hours | Speakers | Utterances (approx.) | 2-s segments per epoch |
|---|---|---|---|---|---|
| **A: licence-clear today** | LibriTTS-R train-clean-100 + train-other-500, VCTK, AMI (minus evaluation speakers) | ≈ 510 | ≈ 1 700 | ≈ 380 k | ≈ 0.92 M |
| **B: plan** | A + Common Voice en (sampled, after decision D1) | ≈ 1 010 | ≈ 6 000+ | ≈ 740 k | ≈ 1.8 M |
| Attacker pool (eval) | LibriSpeech train-clean-360 | ≈ 360 | 921 | ≈ 104 k | — |
| Arabic | **none usable commercially today** (ARABIC_NOT_READY) | 0 | 0 | — | — |

## 3. Steps, sequence length, epochs

* **Batch and sequence:** batch 16 utterance segments of 2.0 s, i.e. 200 frames of
  10 ms and 32 000 samples each.
* **Steps:**

| Stage | Steps |
|---|---|
| 1 | 200 k |
| 2 | 300 k |
| 3 | 200 k |
| 4 (QAT) | 20 k |
| **Total** | **720 k** |

* **Epoch equivalents:** ≈ 12.6 for scenario A, ≈ 6.4 for scenario B.
* **Total training FLOPs:**
  200 k × 0.042 + 300 k × 4.10 + 200 k × 8.3 + 20 k × 4.1 TFLOP ≈ **3.0 × 10¹⁸ FLOP**.
  This assumes a stage-4 step costs about the same as stage 2.

## 4. GPU memory (derived from measured activations)

* **bf16 autocast** roughly halves activations.
* **Not measured:** stage 3 also keeps HuBERT activations for the backward pass through
  ŷ, estimated at +3–5 GB in fp32.

| Precision | Batch | Stage 3 peak (est.) | Fits |
|---|---|---|---|
| bf16 | 16 × 2 s | ≈ 12–14 GB | 24 GB cards (4090, L4, A10G), A100 40/80 |
| fp32 | 16 × 2 s | ≈ 20–23 GB | tight on 24 GB; A100 |
| bf16 | 8 × 2 s + grad-accum 2 | ≈ 7–8 GB | 16 GB cards (T4, V100-16) |

## 5. Throughput and GPU hours

**Derivation:** hours = 3.0 × 10¹⁸ / (peak BF16 dense × MFU) / 3600.

**Peak values:** recalled vendor datasheet values (BF16 dense, no sparsity), **NOT
VERIFIED** in this environment:

| GPU | Peak BF16 dense (TFLOP/s) |
|---|---|
| A100 | 312 |
| RTX 4090 | 165 |
| L4 | 121 |
| A10G | unknown (planning range 70–125) |

**MFU** (achieved fraction of peak) is the dominant unknown. GAN-vocoder training (many
small convolutions, weight-norm, STFTs, data loading) is typically low, so three
scenarios are shown:

| GPU | MFU 10 % | MFU 20 % | MFU 35 % | Memory note |
|---|---|---|---|---|
| A100 80 GB | 26.5 h | 13.3 h | 7.6 h | batch 16–32 bf16 |
| A100 40 GB | 26.5 h | 13.3 h | 7.6 h | batch 16 bf16 |
| RTX 4090 24 GB | 50 h | 25 h | 14 h | batch 16 bf16 |
| L4 24 GB | 68 h | 34 h | 20 h | batch 16 bf16; low memory bandwidth makes the low-MFU column more likely |
| A10G 24 GB | 66–118 h | 33–59 h | 19–34 h | peak NOT_VERIFIED |

**Overheads per full run (estimates):**

| Item | A100-hours |
|---|---|
| Validation rounds (every 5 k steps: 144 rounds × render + Whisper + 2 ASVs) | ≈ 3–6 h |
| Stage 0 teacher units (13.8 GFLOP per second of audio; 1 000 h of audio ≈ 5 × 10¹⁶ FLOP) | < 1 h |
| Attackers: anonymise the 360 h pool + fine-tune / train ECAPA | ≈ 3–8 h |

**Planning figures**

| Scope | A100-hours | RTX 4090-hours |
|---|---|---|
| **One full run** | **≈ 20–40 h** | ≈ 40–70 h |
| **Programme:** 2 full runs + 4 ablations at 1/3 length + attackers + evaluations | **≈ 100–200 h** | ≈ 200–350 h |

These figures are **lower** than the earlier plan (50–65 h per run). That plan was not
based on measured FLOPs. Both remain estimates until measured on the real GPU.

## 6. Checkpoints

| Item | Value |
|---|---|
| Resumable checkpoint (generator + conditioning encoder + discriminators, fp32 + Adam moments) | ≈ 0.27 GB (measured parameter count × 12 B) |
| Frequency | every 5 000 steps (resumable) + stage-exit + best-on-validation |
| Retention | keep the last 3 + best + exits ≈ 10 per run ≈ 3 GB |
| Exported INT8 model | 7.2–7.5 MB (smoke export measured) |
| Storage, scenario A | sources ≈ 100–200 GB + 16 kHz FLAC copies ≈ 30 GB + attacker pool ≈ 25 GB → **< 0.5 TB** |
| Storage, scenario B | ≈ 0.5–1 TB |

## 7. Cost

| GPU | Price / hour | One full run | Programme |
|---|---|---|---|
| RTX 4090 | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED |
| L4 | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED |
| A10G | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED |
| A100 40 GB | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED |
| A100 80 GB | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED | PRICE_NOT_VERIFIED |

Cost = (GPU-hours from §5) × (verified hourly price) + storage + egress. Prices must be
taken from the provider's page on the day of booking and recorded here.

## 8. Minimum vs recommended

| | GPU | Why |
|---|---|---|
| Minimum | 1 × 24 GB (RTX 4090 / L4 / A10G), bf16, batch 16 | fits stage 3 at batch 16 (§4) |
| Recommended | 1 × A100 40/80 GB | about 2× the 4090 throughput; headroom for batch 32 and attacker training |
| Not recommended | 16 GB cards (T4 / P100), Kaggle / Colab free tiers | batch 8 + accumulation, about 4–8× slower; session limits break 20–70 h runs |

The first real-GPU action is a **1 000-step throughput probe** per stage. It replaces
§5's MFU assumption with a measurement before any long run is booked.
