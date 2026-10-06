# StreamAnon: training plan (GPU; not started)

Training has **not** started. CPU training is refused by design: `training/trainers/train.py`
exits with status 2 and lists what is missing. Every number below is a plan or an
estimate unless it is labelled as measured.

Dataset statistics are approximate public figures. The licence column must be
re-checked and accepted by the project owner before download ("verify").

## 1. Stages

| # | Stage | Trains | Losses | Steps (batch 16 × 2 s) |
|---|---|---|---|---|
| 0 | Assets | k-means teacher units (K=500, mHuBERT-147 layer 9); frozen speaker encoders for training; YIN F0 for all audio | — | — |
| 1 | Content distillation | encoder, VQ, unit head, CTC head | unit CE, CTC, VQ commitment | 200 k |
| 2 | Reconstruction | decoder, spectral head, conditioning encoder C | mel L1, MR-STFT, GAN, feature matching, F0 follow | 300 k |
| 3 | Anonymization | all | stage-2 losses (own-speaker half of each batch) + content-output, speaker adversarial, speaker suppression, pseudo-consistency, anti-impersonation, temporal (pseudo-speaker half) | 200 k |
| 4 | QAT INT8 | all, fake-quant | mel, MR-STFT, content-output, speaker suppression | 20 k |

* **Stage 3 mixes both modes in every batch.** Half the items are reconstructed with
  their *own* speaker vector, which keeps a real target waveform for the GAN and mel
  losses. The other half are rendered with a *sampled pseudo-speaker*, where no target
  waveform exists, so content, speaker and attribution losses govern it.
* **Look-ahead:** targets are delayed by 2 frames (`delay_frames`).
* **Streaming:** training uses full sequences with causal layers; streaming equivalence
  is guaranteed by construction and tested.

## 2. Data requirements

| Requirement | Value | Why |
|---|---|---|
| Speakers (training) | ≥ 5 000, of which ≥ 1 500 with ≥ 10 min | The adversary and the pseudo-speaker prior need a dense speaker space. VPC's attacker alone uses 921 speakers. |
| Hours (training) | ≈ 1 500–2 000 h (≈ 1 200 English/multilingual + ≥ 400 Arabic) | A content encoder that holds up on spontaneous, noisy speech |
| Sampling rate | Source ≥ 16 kHz, stored as 16 kHz FLAC | Model rate; 8 kHz telephone data only as augmentation |
| Sessions per speaker | ≥ 2 for every evaluation and attacker speaker | Cross-session attacks |
| Conditions | Read, spontaneous, conversational, broadcast; clean and noisy (SNR 0–30 dB); reverberant; several devices/codecs | The app runs on phone mics in real rooms |
| Splits | Speaker-disjoint `train`, `valid`, `attacker_train`, `test` (`training/datasets/manifest.py` enforces disjointness) | Unseen-speaker evaluation; an attacker pool disjoint from both model training and test |
| Test | ≥ 40 unseen speakers per language, ≥ 2 sessions each, balanced by gender | Our current 9-speaker test gives ±10–15 % EER confidence intervals |

### English / multilingual corpora

| Corpus | Use | Size (approx.) | Style / conditions | Licence |
|---|---|---|---|---|
| LibriTTS-R | train (stages 1–3) | 585 h, 2 456 speakers, 24 kHz | read, clean, restored | CC BY 4.0 |
| LibriSpeech | train (960 h), **attacker** (train-clean-360, as VPC), **test** (test-clean/other) | 960 h, 2 338 + 73 speakers | read audiobooks; chapters act as sessions | CC BY 4.0 |
| VCTK | train / valid | 44 h, 110 speakers, 48 kHz | read, many accents | CC BY 4.0 |
| Common Voice (en, sampled) | train | ~500 h sampled from many thousands of speakers | read, crowd-sourced devices and noise | CC0 |
| AMI Meeting Corpus | train / test (spontaneous) | 100 h | meetings, spontaneous, far/near field | CC BY 4.0 |
| VoxCeleb1/2 | training-time speaker encoders, adversary diversity, **cross-session test** (Vox1-O) | 1 251 + 6 112 speakers, ~2 400 h | in-the-wild interviews (YouTube) | **verify** (research terms; celebrities) |
| GigaSpeech (optional) | spontaneous podcasts/YouTube | 10 000 h available | spontaneous | **verify** (access agreement) |
| Switchboard / Fisher (optional) | conversational telephone | 260 h / 2 000 h | conversational, 8 kHz | LDC paid licence |
| MUSAN | noise/music/babble augmentation | 109 h | — | CC BY 4.0 |
| OpenSLR-28 | room impulse responses | — | — | Apache-2.0 |
| DNS Challenge noise | noise augmentation | — | — | mixed per clip, **verify** |

### Arabic

See `STREAMING_NEURAL_ARABIC_PLAN.md`: Common Voice ar, MASC, MGB-2, QASR, SADA,
FLEURS, and our own consented Jordanian/Levantine recordings.

## 3. Frozen training-time models (never used as evaluators)

| Role | Model | Licence |
|---|---|---|
| Content teacher (units + output content loss) | mHuBERT-147 (multilingual incl. Arabic), layer 9; English-only ablation: HuBERT-base layer 6 | **verify** before use. If the teacher licence is non-commercial, decide whether a distilled student may be shipped (legal question, flagged). |
| Speaker encoders inside losses | WeSpeaker ResNet-34 (VoxCeleb) + an in-house ECAPA trained on VoxCeleb2 + Common Voice | verify / own |
| Differentiable pitch net (F0-follow loss) | small CNN pitch estimator trained on PTDB-TUG/clean data, or an equivalent open model | verify |
| Phone/grapheme CTC targets | English: G2P (CMUdict-based); Arabic: grapheme CTC (no reliable diacritics in dialect) | — |

**Evaluation models are held out:** GE2E, the VPC 2024 ECAPA, SA-toolkit ResNet-Vox1, and
the retrained attackers never appear in any training loss.

## 4. Augmentation (input side only; targets stay clean)

* Additive noise (MUSAN/DNS), SNR uniform 0–30 dB, p = 0.6.
* RIR convolution, p = 0.3.
* Codec simulation (AMR-WB, Opus 16 kbit/s), p = 0.2.
* Gain −12…+6 dB; band-limiting 4–8 kHz, p = 0.1.
* Input passes through the **same noise suppressor as the app** (C++ binding), so
  training matches the runtime front end.

## 5. Pseudo-speaker conditioning in training

* **Conditioning encoder C:** utterance-level, 128-d. It is trained jointly in stage 2 so
  the decoder learns to render *any* vector in C's space.
* **Embedding augmentation in stage 2–3:**
  * interpolate between two speakers' vectors (mixup α ∈ U(0,1));
  * Gaussian noise σ = 0.1 in whitened space;
  * samples from the prior N(μ, Σ).
* This keeps the decoder on-manifold for sampled pseudo-speakers (risk R3).
* Pseudo-speaker F0: each vector carries a median log-F0 and range. The decoder's
  prosody input is z-normalised, so the synthetic level enters only through the vector.

## 6. Losses (implemented in `training/losses/losses.py`; initial weights in `configs/*.yaml`)

Notation:

| Symbol | Meaning |
|---|---|
| x | input |
| ŷ | output |
| s | conditioning vector: own or pseudo |
| E_k | frozen *training* speaker encoders |
| T | frozen content teacher |
| e_j | training-speaker centroids |
| D_d | discriminators |

| Loss | Formula | Mode | Weight |
|---|---|---|---|
| Reconstruction, mel | ‖logmel(ŷ) − logmel(x)‖₁ | own | 45 |
| Reconstruction, MR-STFT | mean over (512,128), (1024,256), (256,64) of SC + ‖log\|S(ŷ)\| − log\|S(x)\|‖₁ | own | 1 |
| Perceptual, adversarial | LSGAN: Σ_d E[(D_d(ŷ) − 1)²]; MPD (periods 2,3,5,7,11) + multi-resolution STFT discriminator | both | 1 |
| Perceptual, feature matching | Σ_d Σ_l ‖D_d^l(x) − D_d^l(ŷ)‖₁ | own | 2 |
| Content, units | CE(unit_head(z_t), u_t), teacher units repeated to 10 ms and delayed 2 frames | both | 1 |
| Intelligibility, CTC | CTC(ctc_head(z), phones/graphemes) | both | 0.5 |
| Content, output | mean_t [1 − cos(T(ŷ)_t, T(x)_t)], teacher layer 9 | pseudo | 10 |
| VQ commitment | ‖z_e − sg(z_q)‖²; codebook EMA, decay 0.99 | both | 0.25 |
| Speaker adversarial | CE(spk_head(GRL_λ(mean_t z, std_t z)), speaker); λ ramps 0 → 1 over 50 k steps (DANN schedule) | both | 1 (λ) |
| Speaker suppression | Σ_k mean relu(cos(E_k(ŷ), E_k(x)) − m), m = 0.25 | pseudo | 2 |
| Pseudo-speaker consistency | [1 − cos(C(ŷ), s)] + within-session spread of C(ŷ) | pseudo | 1 |
| Anti-impersonation | mean relu(max_j cos(E(ŷ), e_j) − τ), τ = p99 of unrelated-speaker cosine under E | pseudo | 2 |
| Temporal consistency | ‖ê_w − mean_w ê‖² over 1.5 s windows of one output | pseudo | 0.5 |
| F0 follow | ‖z-logF0(ŷ) − z-logF0(x)‖₁ on voiced frames (frozen pitch net) | both | 1 |

Weights are starting points.
* **Tuned on VALID speakers only**, with a fixed budget of at most 8 runs of 20 k steps.
* **Selection rule:** maximise validation lazy-informed EER subject to Whisper relative
  WER ≤ 15 %.
* Test speakers are never used for tuning.

## 7. Training compute (estimates)

Assumptions:
* **Generator:** 6–11 M params.
* **Discriminators:** HiFi-GAN-style MPD plus MR-STFT discriminator, ≈ 40–70 M params.
* **Frozen models in stage 3:** mHuBERT (~95 M) and two speaker encoders.
* **Precision:** bf16.
* **Batch:** 16 × 2 s.

| Item | A100 80 GB | RTX 4090 24 GB | T4 16 GB |
|---|---|---|---|
| Stage 0 (teacher units + F0 for ~2 000 h) | ~4–6 h | ~6–8 h | ~25–35 h |
| Stage 1 (200 k steps, ~12 steps/s) | ~5 h | ~6 h | ~25 h |
| Stage 2 (300 k steps, ~4–5 steps/s) | ~17–21 h | ~20–26 h | ~4–5 days |
| Stage 3 (200 k steps, ~2–3 steps/s) | ~19–28 h | ~23–34 h | ~5–7 days |
| Stage 4 QAT (20 k) | ~2 h | ~2–3 h | ~10 h |
| **One full run** | **≈ 50–65 GPU-h (≈ 2.5 days)** | **≈ 60–80 GPU-h** | **≈ 11–15 days** |
| Attackers (VPC-ECAPA fine-tune + scratch ECAPA) | ~3 h + ~15 h | ~4 h + ~20 h | impractical |
| Programme (1 full run, 3–5 shortened ablations, 2nd full run, attackers) | **≈ 150–250 GPU-h (6–10 days)** | ≈ 200–300 GPU-h | not recommended |

| Resource | Estimate |
|---|---|
| VRAM at batch 16 × 2 s | ≈ 14–20 GB (discriminators on waveform dominate; frozen teacher ≈ 2 GB) |
| VRAM option | batch 8 + gradient accumulation 2 fits 16 GB, about 1.5× slower |
| **Minimum GPU** | 1 × 24 GB (RTX 3090/4090, L4, A10G) |
| **Recommended GPU** | 1 × A100 40/80 GB, or L40S |
| Checkpoints | generator 24–44 MB (fp32); full resumable checkpoint (+ discriminators + Adam moments) ≈ 0.6–1.0 GB; keep last 3 + best ≈ 4 GB per run |
| Exported model | INT8 7–13 MB |
| Storage | 0.5–1 TB SSD for source corpora (VoxCeleb2 and Common Voice dominate) + 16 kHz FLAC copies + units (≈ 0.5 GB) |

**Colab / Kaggle:**
* Kaggle (≈ 30 GPU-h per week on T4/P100) is enough for stage-1 sanity runs on a 100 h
  subset, not for the programme.
* Colab Pro+ with A100 works only with checkpoint/resume across sessions, and keeping
  hundreds of GB of data mounted is the real obstacle.
* A rented single-GPU VM with local SSD is recommended.
* All of this concerns **training only**. The app's runtime stays on-device, with no
  cloud processing.

## 8. Commands

```bash
# manifests (one JSONL per split; audio never in git)
python3 training/datasets/manifest.py validate data/manifests/train.jsonl
# stage 0
python3 training/scripts/compute_teacher_units.py --manifest data/manifests/train.jsonl \
    --teacher models_train/mhubert-147 --layer 9 --k 500 --out data/teacher
# training (GPU)
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --out runs/s
# pseudo-speaker pool (after training)
python3 training/scripts/make_pseudo_pool.py --centroids runs/s/train_speaker_centroids.npy --n 10000 --out build/pseudo_pool.npy
# export + INT8
python3 training/export/export_onnx.py --config training/configs/stream_anon_s.yaml \
    --checkpoint runs/s/qat_int8/generator.pt --out build/stream_anon_s.onnx
python3 training/quantization/quantize_int8.py --in build/stream_anon_s.onnx --out build/stream_anon_s.int8.onnx --mode dynamic
# evaluation and benchmarks: see STREAMING_NEURAL_EVALUATION_PLAN.md and STREAMING_NEURAL_ANDROID_PLAN.md
```

## 9. Training loop

The loop is not committed as executable code before the assets exist, so nothing in this
repository can produce an untrained "result". It is written on the GPU machine against:
* the fixed interfaces: model, losses, manifest, config;
* standard HiFi-GAN-style discriminators;
* an EMA VQ codebook;
* a bf16 autocast loop with two optimisers.

## 10. Risks and mitigations

| Risk | Mitigation / fallback |
|---|---|
| R1 Privacy plateau against the semi-informed attacker (rhythm/contour leakage) | Optional contour quantisation (`causal_prosody(bins=…)`); stronger adversary weight; report honestly. If EER < 25 %, **fail** the gate. |
| R2 Intelligibility (20 ms look-ahead, 6 M params) | M config; look-ahead 3–4 frames (+10–20 ms); stronger CTC weight; teacher layer choice |
| R3 Off-manifold pseudo-speakers | Embedding augmentation; prior-sampled training; vetting step 3; backup StreamKNN |
| R4 INT8 quality loss | Per-channel static QDQ; keep head/FiLM in FP16 (+1–2 MB) |
| R5 Over-fitting to training speaker encoders (Goodhart) | Two different training encoders; held-out evaluators; retrained attackers |
| R6 Noise robustness | App NS before the model; noise augmentation; noisy test subsets |
| R7 Licences (VoxCeleb, teacher, Arabic corpora) | Owner review before download; CC-licensed-only fallback configuration |

## 11. What is actually needed to start training

1. A GPU machine: ≥ 1 × 24 GB GPU (recommended A100-class), ≥ 1 TB SSD, ~250 GPU-hours.
2. Licence decisions:
   * VoxCeleb, mHuBERT-147 (or another teacher), WeSpeaker;
   * the Arabic corpora: MASC, MGB-2/QASR, SADA.
3. Downloaded corpora and validated manifests (§2).
4. Frozen training-time models (§3) under `models_train/`.
5. The app's YIN pitch tracker and noise suppressor as a Python extension. This is
   small work that can be done without a GPU and is the first engineering task.
6. Consented Arabic (Jordanian/Levantine) recordings for validation and test
   (`STREAMING_NEURAL_ARABIC_PLAN.md`).
7. An Android device to run the untrained-export compute benchmark
   (`STREAMING_NEURAL_ANDROID_PLAN.md` §7). That can happen **before** training.
