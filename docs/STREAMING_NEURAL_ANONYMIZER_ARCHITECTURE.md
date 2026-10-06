# Streaming neural voice anonymizer: architecture (design phase)

**Status: DESIGN + UNTRAINED PROTOTYPE.** Nothing has been trained. No anonymization,
quality or Android result is claimed. The prototype in `training/` is used only for
graph tests (shapes, causality, streaming state, export, INT8) and for compute-cost
measurements; its weights are random.

**Decision gate: B — FEASIBLE WITH CHANGES** (section 9). The DSP engine and the Android
app are unchanged.

Companion documents:
* `STREAMING_NEURAL_TRAINING_PLAN.md`: data, losses, stages, GPU.
* `STREAMING_NEURAL_EVALUATION_PLAN.md`: attackers, metrics, acceptance criteria.
* `STREAMING_NEURAL_ANDROID_PLAN.md`: runtime integration and the DSP fallback.
* `STREAMING_NEURAL_ARABIC_PLAN.md`.

## 1. What the measured evidence says the design must do

| Evidence (this repo, 9 unseen speakers, informed attacker, cross-session) | Consequence for the design |
|---|---|
| DSP Strong (pitch/formant/spectral): EER 1.2–1.3 %, top-1 96–100 % under VPC-ECAPA / ResNet | Transforming the voice keeps every speaker separable. The design must **replace** the speaker, not transform it. |
| WORLD pseudo-speaker (replace F0 level + mean envelope): EER 2.6–9 % | Replacing a few long-term statistics is not enough. Identity also lives in frame-level spectral detail and its dynamics. |
| kNN-VC (every frame replaced by frames of a synthetic voice): EER 28–36 %, top-1 26–41 %, Whisper WER 34 %, 332 M params, 3 GB, non-causal | Frame-level replacement works for privacy. A bank of only ~29 s of target frames loses content. Its size and non-causality rule it out on a phone. |
| VPC B3 (phones + GAN artificial speaker + TTS): EER 41–51 % here (official semi-informed attacker 22–28 %), WER 45 % | (a) A narrow bottleneck plus an artificial speaker gives the strongest privacy, but hard phone decisions destroy content. (b) A **GAN/prior-sampled artificial speaker** was the least similar to any real person (attribution 0.105–0.154, below the unrelated-voice level). (c) A strong attacker still recovers about a quarter. |
| Target attribution | Pseudo-speakers sampled from a prior and vetted against real speakers do not impersonate anyone; this check has to be part of the design. |

Design consequence: a **causal content bottleneck** with frame-rate and soft content (not
hard phones), plus **active speaker removal** in that bottleneck, plus **resynthesis**
with a **vetted synthetic speaker vector**. It must be small enough for one phone core.

## 2. Chosen architecture: StreamAnon (primary)

```
mic 48 kHz ─▶ [existing DSP front: DC/HPF + noise suppressor + YIN pitch tracker]
          ─▶ 48→16 kHz polyphase resampler
          ─▶ causal log-mel 80 (20 ms sqrt-Hann window ENDING at the current sample, 10 ms hop)
          ─▶ ContentEncoder: causal conv k3 → 6 × causal ConvNeXt (dim 192, dw-k7, MLP×3) → GRU 192
          ─▶ Bottleneck: Linear 192→64 → VQ (512 codes)          ◀── speaker removal (§3)
  prosody: YIN F0 → log-F0 z-scored by a causal running mean/var (absolute level removed),
           voicing flag, causally normalised energy → Linear 3→32
          ─▶ Decoder: Linear (64+32)→320 → 6 × causal ConvNeXt (dim 320) with FiLM(pseudo-speaker 128-d)
          ─▶ Head: Linear 320→2×161 = log-magnitude + phase per bin (Vocos-style)
          ─▶ inverse rFFT 320 + sqrt-Hann overlap-add (10 ms hop)
          ─▶ 16→48 kHz resampler ─▶ [existing look-ahead limiter] ─▶ out
```

| Item | Value |
|---|---|
| Sample rate inside the model | 16 kHz (wideband; content up to 8 kHz). See the bandwidth note below. |
| Input representation | 80 log-mel (0–8 kHz) + 3 prosody features per 10 ms frame |
| Frame / hop | 20 ms analysis window, 10 ms hop |
| Context | Encoder receptive field: 2 + 6×6 = 38 frames (380 ms) of past through convolutions, plus unbounded past through the GRU. Decoder: 36 frames of past. |
| Causality | Every layer is causal. A 2-frame (20 ms) look-ahead is **learnt by delaying the training target**, so the graph has no future input and needs no special streaming code. |
| Encoder | 1.63 M params (S) |
| Bottleneck | 64-d, VQ 512 codes (9 bits per 10 ms ≈ 900 bit/s: room for phonetic content, little for identity) |
| Speaker removal | VQ bottleneck + unit/CTC content supervision + gradient-reversal speaker adversary + output-level speaker suppression + anti-impersonation (§3) |
| Pseudo-speaker | 128-d vector drawn per session from a vetted pool, applied by FiLM in every decoder block; it also carries the synthetic F0 level and range (§4) |
| Decoder / vocoder | 4.35 M params (S), incl. FiLM and the spectral head. A single causal network predicts the STFT directly, so there is no mel→waveform vocoder stage and no transposed convolutions. |
| Streaming state | 14 tensors: the input-conv history (2 frames × 80), 12 depthwise-conv histories (6 frames each) and the GRU hidden state; 18 784 floats ≈ 75 KB. |
| Overlap-add | Yes: sqrt-Hann 320 / hop 160, perfect reconstruction (tested). Completing each output hop needs the next frame (10 ms). |
| Quantisation | INT8 weights (dynamic, measured); static QDQ INT8 activations after training with calibration; head may stay FP16 if INT8 hurts quality |

### Measured on the untrained prototype

These numbers are compute only, measured on a 4-core Xeon 2.1 GHz with ONNX Runtime
1.30 CPU on 1 thread (`docs/results/streaming_prototype_bench.json`).

| Config | Deployable params | MAC / s | ONNX FP32 | ONNX INT8 | Step time, K=1 (10 ms) | RTF K=1 | RTF K=2 (20 ms chunks) | p99 RTF K=2 |
|---|---|---|---|---|---|---|---|---|
| **S** (primary) | **6.00 M** | **0.55 G** | 24.2 MB | **7.0 MB** | 1.29 ms | 0.129 | **0.076** | 0.118 |
| M (if S under-fits) | 10.97 M | 1.01 G | 44.0 MB | 12.6 MB | 1.79 ms | 0.179 | 0.109 | 0.156 |

* **Per-call overhead dominates at K=1.** About 100 small operators are dispatched per
  call, so processing 2 frames per call almost halves the cost. **K=2 is the operating
  point**, at the cost of +10 ms latency.
* **Process RSS of the host benchmark:** 75–83 MB, including the Python interpreter and
  ORT. The model itself (INT8) is 7–13 MB, and its activations and state are < 2 MB.

### Expected Android cost (INTERPRETATION, not measured: no device)

A mid-range big core (Cortex-A76/A78 at 2.2–2.4 GHz, int8 dot-product) is about 2–4×
slower than this Xeon core for small batch-1 GEMV/GEMM:

| Config | Expected RTF on 1 big core | Added memory |
|---|---|---|
| S, K=2 | **≈ 0.15–0.30** | ≈ 25–40 MB (ORT runtime + 7 MB model + buffers) |
| M, K=2 | ≈ 0.22–0.44 | ≈ 35–50 MB |

* Little cores (Cortex-A55) are 3–5× slower again: the model must run on a big core.
* Phones with only little cores fall back to DSP.
* The first device benchmark will replace these estimates. The procedure is in
  `STREAMING_NEURAL_ANDROID_PLAN.md`.

### Algorithmic latency

| Component | K=1 | K=2 (chosen) |
|---|---|---|
| Collecting the input hop / chunk | 10 ms | 20 ms |
| Learnt look-ahead (delay_frames = 2) | 20 ms | 20 ms |
| Overlap-add completion (win − hop) | 10 ms | 10 ms |
| **Algorithmic total** | **40 ms** | **50 ms** |
| 48↔16 kHz resampling (two linear-phase polyphase FIRs) | +2–3 ms | +2–3 ms |
| Compute deadline margin (one call's worst case) | + ≤ 10 ms | + ≤ 10 ms |

Today's DSP has 32 ms algorithmic latency. Expected mic→ear on a good AAudio device is
≈ 80–110 ms, against ≈ 60–80 ms for the DSP.

### Bandwidth note

The model works at 16 kHz to fit the budget, so the output has no energy above 8 kHz:
wideband-call quality, not studio quality. A 24 kHz variant costs about +50 % compute
(n_fft 480, hop 240 at the same 10 ms, plus a larger head). It is an ablation, not the
default.

## 3. Speaker-identity removal: options studied and the choice

| Option | Mechanism | Verdict |
|---|---|---|
| A content/speaker disentanglement | Separate content and speaker paths, reconstruct from content + speaker | **Used** as the backbone: the encoder sees no speaker vector, and the decoder gets only the pseudo-speaker |
| B adversarial speaker classifier | Gradient reversal: a speaker classifier on utterance statistics (mean/std) of the bottleneck; the encoder maximises its loss | **Used.** It acts on exactly what a speaker-ID attacker pools. Ramped in after content is learnt. Alone it is known to be incomplete: adversaries are fooled locally while information remains. |
| C speaker-embedding suppression | Penalise cos(E_k(ŷ), E_k(x)) > m under frozen *training* speaker encoders | **Used** at the output, against ≥ 2 encoders that are never used for evaluation (to avoid Goodhart) |
| D pseudo-speaker generation | Sample a synthetic speaker vector from a prior fitted to training speakers; reject vectors close to any real speaker | **Used.** B3's GAN prior was the least attributable to real people in our test |
| E random synthetic conditioning per session | New vector per call/session, fixed within it | **Used.** It also defeats cross-session linking (the A→B attacks) |
| F VQ / information bottleneck | Discrete 64-d codes at 10 ms, 512 entries | **Used.** It caps the information rate. B3 shows a hard phone bottleneck removes identity but also content, so codes are trained with unit + CTC supervision and kept at 10 ms with 512 entries (not ~40 phones). |
| G neural resynthesis | The whole waveform is regenerated from the bottleneck | **Used** (Vocos-style STFT head). This removes source excitation and fine structure. |
| H combination | — | **Chosen: A + B + C + D + E + F + G** |

Rationale against the measured systems:
* **DSP/WORLD fail** because they are deterministic per-speaker transforms.
* **kNN-VC succeeds on privacy** because it replaces frames. It loses content because
  frame replacement from a 29 s bank cannot express all phones. A learnt decoder
  conditioned on a pseudo-speaker vector can render *any* content in *any* sampled voice,
  so there is no bank-coverage problem.
* **B3** proves the narrow-bottleneck + artificial-speaker principle. Its WER (45 %) and
  latency come from ASR→TTS with hard phones, which StreamAnon avoids.

Residual leakage that this design **cannot** remove in real time:
* **Speaking rate and rhythm.** Timing passes through 1:1, because changing durations
  needs future context.
* **Intonation contour shape** after z-normalisation; optional quantisation coarsens it.
* **Lexical/linguistic habits.**

This is why the privacy targets in the evaluation plan are set **below** "chance". A
semi-informed attacker can use these cues.

### Backup architecture: StreamKNN

This is the streaming analogue of the measured kNN-VC.

```
causal student encoder (distilled from WavLM-Large layer 6, the feature kNN-VC uses; ~4–6 M params)
  → kNN regression (k=4, cosine) against an on-device SYNTHETIC matching bank:
    2 048 k-means centroids × 256-d int8 (0.5 MB), one bank per pseudo-voice
  → causal STFT-head vocoder trained on "prematched" features (as kNN-VC's prematched HiFi-GAN)
```

* **Cost:** ≈ 0.5 M MAC/frame for matching plus the encoder and vocoder (≈ 0.6–0.9 GMAC/s).
* **Pro:** it follows the measured kNN-VC result directly, and the voice is a fixed
  synthetic bank, so there is no off-manifold conditioning.
* **Con:** the synthetic banks must come from a generator, i.e. the primary model or a
  TTS with GAN speakers. Content coverage of a small bank is the measured weakness.
* **Use it if** StreamAnon's decoder cannot render sampled pseudo-speakers naturally
  (risk R3 in the training plan).

### Rejected

* **ASR→TTS** (B3-style) on device: utterance latency and WER.
* **Real-target VC** (LLVC, B5/B6): impersonation, forbidden.
* **Porting WavLM-Large or kNN-VC:** 332 M params / 3 GB.
* **Stronger DSP:** measured insufficient.
* **NNAPI:** deprecated from Android 15.
* **GPU delegate:** per-10-ms dispatch overhead and power.

## 4. No impersonation: pseudo-speaker mechanism and attribution test

1. After training, embed every TRAIN speaker with the conditioning encoder C (centroids
   e_j). Fit N(μ, Σ) to the centroids (`training/models/pseudo_speaker.py`).
2. Sample candidates s and **reject** any candidate with:
   * max_j cos(s, e_j) > τ_attr, where τ_attr = 99th percentile of cosine similarity
     between *different* real training speakers; or
   * Mahalanobis distance above the 95th percentile of real speakers (implausible voice).
3. Render each survivor on a fixed validation set and keep it only if the **output** also
   passes, under each frozen training speaker encoder:
   * max_j cos(E(ŷ), e_j) ≤ τ_attr(E), with no real training speaker closer than unrelated
     speakers usually are;
   * Whisper WER and DNSMOS within limits.
4. Ship only the surviving pool: 10 000 × 128 int8 ≈ 1.3 MB, with each vector's F0
   median and range. Training-speaker centroids are never shipped.
5. **On the device:**
   * At the start of each session/call, draw one pool index with a CSPRNG.
   * Never reuse one of the last 50 indices.
   * The voice is fixed for the session, so it stays natural and consistent within a
     call, and a new call gets a new voice.
   * Only indices are stored, never audio.

**Attribution test, run at evaluation.** It extends `scripts/target_attribution.py`.
For processed TEST speech, compute the best match to every TRAIN speaker and every
speaker in the corpora used to fit the prior, under the *evaluation* ASVs. Pass only if:
* the mean best-match similarity is ≤ the 95th percentile of the unrelated-real-voice
  reference; and
* < 1 % of utterances exceed the **median** same-speaker similarity.

The Phase 3 attribution tool used a 5th-percentile threshold. That proved too loose (it
sits below the unrelated level), so the median is used here.

## 5. Interfaces fixed now (so training, evaluation and Android agree)

* **Frames:** `frontend.causal_stft` / `CausalLogMel` / `istft_ola` are the reference;
  the C++ runtime reimplements them with the existing `dsp/` FFT and must match within
  1e-4.
* **F0:** the app's C++ YIN (`dsp/src/pitch_tracker.cpp`) is the runtime tracker.
  Training must use the same code through a Python binding.
* **Prosody normalisation:** `frontend.causal_prosody`, a cumulative-then-EWMA log-F0
  mean/variance, is exactly invariant to a constant pitch-level shift from the first
  voiced frame (tested).
* **ONNX step:**
  * inputs `mel [1,K,80]`, `prosody [1,K,3]`, `spk [1,128]`, `state_0..13`;
  * outputs `spec [1,K,161,2]`, `new_state_0..13`;
  * metadata hop/win/sr/delay_frames/untrained (`training/export/export_onnx.py`);
  * exported streaming output matches PyTorch within 1e-3 (tested).

## 6. Prototype tests (`python3 training/tests/test_architecture.py`, 26 tests)

All 26 tests pass:
* STFT/iSTFT perfect reconstruction; causal mel; prosody level invariance.
* Output shapes and state shapes.
* Causality (a future change does not alter past outputs).
* Streaming equals full-sequence for chunk sizes 1/2/3/8; state carries history.
* Determinism; the conditioning vector changes the output.
* Gradient reversal reaches the encoder.
* Parameter/MAC/latency budgets.
* ONNX streaming parity with PyTorch; INT8 dynamic quantisation runs, is < 0.4× the size
  and < 30 MB.
* The renderer refuses untrained models and compensates the delay.
* Every loss is finite and differentiable; anti-impersonation behaves as specified.
* The pseudo-speaker pool rejects near-real candidates; session draws avoid recent ones.
* Manifest validation (disjoint splits, licence entries); config validation.
* The trainer refuses to start without a GPU and assets.

None of these tests says anything about privacy or audio quality.

## 7. Model size, RAM, CPU summary

| | S (primary) | M | Budget asked | Assessment |
|---|---|---|---|---|
| INT8 size | 7.0 MB | 12.6 MB | 10–30 MB, max 50 | **meets** |
| Added RAM on device | ≈ 25–40 MB (est.) | ≈ 35–50 MB (est.) | "no 3 GB" | **meets** |
| Compute | 0.55 GMAC/s | 1.01 GMAC/s | — | small, but frame-rate GEMV is memory-bound; measured host RTF 0.076 (K=2) |
| RTF, mid-range phone, 1 big core | 0.15–0.30 (est.) | 0.22–0.44 (est.) | < 0.5 | **plausible, unmeasured** |
| Algorithmic latency | 50 ms (K=2) / 40 ms (K=1) | same | ≤ 50 ms | **meets at K=2**; device I/O adds 30–60 ms |
| Streaming / causal | yes | yes | yes | **meets** |

## 8. Main risks (detail and mitigations in the training plan)

* **R1 Privacy plateau.** Against a semi-informed attacker, rhythm and contour leakage
  plus a small causal model may plateau at EER 20–30 %, like B3 (22–28 %).
* **R2 Intelligibility.** 20 ms look-ahead and 6 M params may cost more WER than DSP;
  published streaming VC systems use 60–70 ms.
* **R3 Off-manifold pseudo-speakers.** The decoder may render sampled vectors unnaturally.
  The backup architecture covers this.
* **R4 INT8 degradation** of the spectral head.
* **R5 Device variance and thermals.**
* **R6 Goodhart.** Training against speaker encoders over-states privacy measured with
  similar encoders, hence held-out evaluators and a retrained attacker.
* **R7 Data licences,** especially Arabic, VoxCeleb and the content teacher.

## 9. Decision gate: B — FEASIBLE WITH CHANGES

The architecture, data, training compute and Android inference budget are coherent and
partly measured: size, latency and host compute. Training can start once the items in
`STREAMING_NEURAL_TRAINING_PLAN.md` §11 exist. It is not **A**, because:

1. The privacy target must be **changed**. EER ≥ 35 % against a *strong (fine-tuned)*
   attacker is not a realistic acceptance bar for a causal real-time model when the best
   offline VPC baseline gets 22–28 %. The plan uses ≥ 25 % (minimum) / ≥ 30 % (target)
   against that attacker, and ≥ 35 % against lazy-informed pretrained ASVs.
2. Latency must be **50 ms algorithmic** (K=2), not ≤ 40 ms, for the CPU budget. Output
   bandwidth is **16 kHz**.
3. Device support must be **gated**: at least one ARMv8.2+ big core, with DSP fallback
   on other devices.
4. **Phone RTF is estimated, not measured.** A device benchmark of the untrained export
   is the first thing to run on hardware; it needs no training.
5. **Arabic needs licensed multi-speaker data plus consented Levantine/Jordanian
   recordings** that do not exist yet. Until then: ARABIC_NOT_VERIFIED.

It is not **C**: no budget item is out of reach on the evidence available.
