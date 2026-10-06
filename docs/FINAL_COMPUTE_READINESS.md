# Final training resource + data readiness audit

**Date:** 2026-10-06.

**Verdict: NO-GO.** The machine-readable verdict is `training/readiness.json`, which the
orchestrator enforces. No training, GPU spend, dataset download, model release or Android
integration happened.

**Evidence rule.** **Search result ≠ official evidence.**
* `VERIFIED_OFFICIAL` means the official text was read directly in this phase.
  * GitHub docs at commit `45a0f053ac67e8d1f56fc8f7ee38f0b2a58925c3` of `github/docs`;
  * Hugging Face docs at `a685082860fb06c99c299bb61079a3636adff7d2` of
    `huggingface/hub-docs`;
  * oracle.com and cloud.google.com pages fetched with sha256 recorded.
* Everything else is `SEARCH_ONLY`, `NOT_VERIFIED`, `EXPIRED` or `APPLICATION_REQUIRED`.

**A GPU counts as available only if all four hold:**
1. the source is official;
2. the free/credit status is proven;
3. the commercial-training status is proven;
4. the eligibility is clear.

**No acceptance criterion was changed.** Only the compute strategy changed.

## 1. GitHub Actions (official)

| Key | Value | Evidence (github/docs @45a0f05) |
|---|---|---|
| **GITHUB_GPU_AVAILABLE** | **NO for this account.** GPU larger runners require the **Team or GitHub Enterprise Cloud** plan (organization). Y1X0 is a personal account (`get_me`). | `content/actions/concepts/runners/github-hosted-runners.md`: "if you are on the Team or GitHub Enterprise Cloud plan, you can provision a runner … powered by a GPU" |
| **GITHUB_GPU_FREE** | **NO.** "Larger runners are always charged for, even when used by public repositories or when you have quota available from your plan"; not eligible for included minutes. A payment method is required. | `content/billing/concepts/product-billing/github-actions.md`; `content/actions/concepts/runners/larger-runners.md` |
| **GITHUB_GPU_CREDITS** | **NONE found.** No GPU or larger-runner credits are mentioned in the Actions billing docs or the Education docs. | same + Education docs below |
| **GITHUB_GPU_COST** | **Linux 4-core GPU (Tesla T4, 16 GB VRAM, 28 GB RAM, 176 GB SSD): $0.052/min = $3.12/h**; Windows $0.102/min | `content/billing/reference/actions-runner-pricing.md`; `content/actions/reference/runners/larger-runners.md` |
| **GITHUB_STUDENT_BENEFIT** | Verified students: Copilot; **Codespaces up to 180 core-hours/month (CPU)**; partner offers on education.github.com/pack (unreachable: SEARCH_ONLY). **No GPU and no larger-runner credits from GitHub.** | `content/education/about-github-education/github-education-for-students/about-github-education-for-students.md`, `data/reusables/education/student-codespaces-benefit.md` |
| Standard runners as orchestrator | **YES, free.** "Actions usage is free for … public repositories that use standard GitHub-hosted runners". Private repos get 2,000 min/month on Free. | `content/billing/concepts/product-billing/github-actions.md`, `data/reusables/billing/actions-included-quotas.md` |
| **GITHUB_EVIDENCE_URL** | `https://github.com/github/docs/tree/45a0f053ac67e8d1f56fc8f7ee38f0b2a58925c3/content/actions` (+ `/content/billing`, `/content/education`, `/data/reusables`) | — |

A public repository does **not** get free GPU runners.

## 2. Free GPU providers, re-audited

| Provider | Status | Free amount (as read) | Commercial training | Counted? |
|---|---|---|---|---|
| GitHub GPU runner | VERIFIED_OFFICIAL | none (paid $3.12/h; Team/Enterprise only) | YES (paid service) | no (not free; not available) |
| GitHub standard runners | VERIFIED_OFFICIAL | free in public repos | n/a (CPU) | orchestrator only |
| GitHub Student Pack | VERIFIED_OFFICIAL | Codespaces 180 core-h/month (CPU) | UNKNOWN | no GPU |
| HF ZeroGPU | VERIFIED_OFFICIAL | 5 GPU-min/day (free account) | UNKNOWN (demo service) | no (not training) |
| HF Spaces paid hardware | VERIFIED_OFFICIAL | none. Prices: T4 $0.40/h, L4 24 GB $0.80/h, A10G 24 GB $1.00/h, **A100 80 GB $2.50/h** | UNKNOWN (training in Spaces not covered by the docs read) | no |
| Oracle Cloud Free Tier | VERIFIED_OFFICIAL | US$300 / 30 days (select countries); one account per person | UNKNOWN; GPU eligibility of the trial not stated | no |
| Oracle for Research | APPLICATION_REQUIRED | by award | UNKNOWN | no |
| Google Cloud research credits | APPLICATION_REQUIRED (official: "Apply for up to $5,000") | if awarded | UNKNOWN (search says no) | no |
| Google Cloud $300 Free Program | NOT_VERIFIED (terms page 403) | $300 | UNKNOWN | no |
| Kaggle | SEARCH_ONLY | ~30 GPU-h/week (2×T4 / P100) | UNKNOWN | no |
| Colab free | SEARCH_ONLY | dynamic | UNKNOWN | no |
| Lightning AI | SEARCH_ONLY | ~80 GPU-h/month | UNKNOWN | no |
| Modal | SEARCH_ONLY | $30/month | UNKNOWN | no |
| Azure for Students | SEARCH_ONLY | $100; no GPU quota | UNKNOWN | no |
| AWS research credits | APPLICATION_REQUIRED (page unreachable) | by proposal | UNKNOWN | no |
| NVIDIA Academic Grant | APPLICATION_REQUIRED (unreachable) | ≤ ~30k H100-h | UNKNOWN | no |
| RunPod research/startup | APPLICATION_REQUIRED (unreachable) | ≤ $25k / $1k | UNKNOWN | no |
| Vast.ai startup | APPLICATION_REQUIRED (unreachable) | ≤ $2.5k | UNKNOWN | no |
| Lambda grant | EXPIRED? (Dec 2024 press only) | ≤ $5k | UNKNOWN | no |
| Paperspace free | SEARCH_ONLY | M4000 8 GB | UNKNOWN | no (8 GB) |
| Google TRC | APPLICATION_REQUIRED (unreachable) | TPU allocation | UNKNOWN | no (TPU port needed) |

Details are in `training/gpu_profiles.json`, where `counted_as_guaranteed_free` is false for
every provider.

## 3. Commercial training legality per provider

No provider's terms were read that **explicitly allow** using free GPU or credits to train a
model for a commercial product. Every free source is **COMMERCIAL_TRAINING = UNKNOWN**, so
none is treated as usable.

Paid services (GitHub larger runners, HF paid hardware) are ordinary paid compute. They are
not free, and their terms on training were not read beyond pricing.

## 4. What a T4 16 GB can do (measured memory + analysis)

**Measurement.** `training/scripts/measure_activation_memory.py` records the bytes saved for
backward on CPU with the real StreamAnon-S and full-size discriminators. Raw output:
`docs/results/activation_memory_measured.json`.

| Stage step | fp32 (MB) b=2 / b=4 | half (MB) b=2 / b=4 | per-sample half (MB) | **b=16 half (GB)** | b=16 fp32 (GB) |
|---|---|---|---|---|---|
| 1 content | 116 / 141 | 60 / 74 | 7 | **0.16** | 0.27 |
| 2 reconstruction | 1,565 / 2,764 | 961 / 1,652 | 346 | **5.66** | 9.72 |
| 3 anonymisation (placeholders for frozen encoders) | 2,606 / 4,259 | 1,518 / 2,442 | 462 | **7.80** | 13.84 |

**Two engineering defects were found by this audit and fixed**, both tested:
1. The trainer had **no mixed precision at all**, although the config said bf16. It now
   implements `precision: auto`:
   * bf16 on A100/L4/4090;
   * **fp16 + GradScaler on T4/P100**, with separate generator and discriminator scalers
     stored in checkpoints;
   * CPU runs stay fp32, so smoke results stay bit-exact.
2. Two ops broke under autocast: the dead-code restart (dtype) and iSTFT (complex half).
   iSTFT and the complex spectrum now always run in fp32.

**T4 memory budget, stage 3, batch 16, fp16** (usable ≈ 15 GB):

| Item | GB |
|---|---|
| Activations (measured) | 7.8 |
| Real frozen speaker-encoder activations (backward through ŷ; estimate) | + 1–2 |
| Params + grads + Adam (22.2 M trainable, measured) | + 0.33 |
| CUDA context + cuDNN workspace + fragmentation (assumption) | + 1.5–2.5 |
| **Total** | **≈ 10.7–12.7** |

* This **fits without gradient accumulation**, with a maximum batch of ≈ 20–24.
* fp32 at batch 16 (≈ 16.6–18.6 GB) does **not** fit; batch 8 does.
* Activation checkpointing (discriminators) is a reserve, not needed.

| Stage | T4 16 GB? | How | A100-h | T4-h (≈ ×4.8, peak-ratio; MFU unmeasured) |
|---|---|---|---|---|
| 0 smoke | **yes** (also CPU) | — | 0 | 0 |
| 1 teacher units (Whisper-small, offline) | **yes** | fp16 inference; measured RTF 0.03 on 4 CPU threads, so even CPU works | 2–4 (10 % MLS) | 10–20 |
| 1 content distillation | **yes** | batch 16 fp16 (0.2 GB activations) | 1–3 | 5–15 |
| 2 small-scale training | **yes** | batch 16 fp16, no accumulation | 1.1–2.3 | 5–11 |
| 3 ablations (≤ 8 × 20k steps) | **yes** | batch 16 fp16 | 4.8–9.5 | 23–46 |
| 4 full training | **memory yes; time is the constraint** | batch 16 fp16; fp16 GAN stability to be confirmed in stage 2 (`ABORT` rules active) | 11.4–22.9 | 55–110 |
| 5 QAT/INT8 | **yes** | — | 0.4–0.7 | 2–4 |
| 6 attackers + locked final evaluation | **yes** (ECAPA training batch adjusted) | — | 4–10 | 20–50 |
| Support: in-house speaker encoders (3) | **yes** | — | 23–56 | 110–270 |
| Validation, checkpoint tests | **yes** | — | incl. | incl. |
| **Total (× 1.3 contingency)** | | | **62–141** | **≈ 300–680** |

## 5. MLS English: numbers resolved from the official card

| Field | Value |
|---|---|
| Dataset version | MLS v1, English (OpenSLR SLR94). Statistics and licence from the publisher's card `facebook/multilingual_librispeech`, last updated 2024-08-12. |
| Official source | https://www.openslr.org/94/ (official distributor; **egress-blocked here**); publisher card on Hugging Face (`facebook` org) |
| Official licence | "Public Domain, Creative Commons Attribution 4.0 International Public License (CC-BY-4.0)", card Licensing Information (E1) |
| Exact reported hours | **train 44,659.74 h; dev 15.75 h; test 15.55 h; total 44,691.04 h** |
| Exact reported speakers | **train 5,490 (2,742 M / 2,748 F); dev 42 (21/21); test 42 (21/21); total 5,574** |
| Download manifest hash | **NOT_AVAILABLE.** The official English archive cannot be reached from here, and **the publisher's HF repo does not contain English** (configs: dutch, french, german, italian, polish, portuguese, spanish only). The HF mirror `parler-tts/mls_eng` (E1-m: 10.8 M train rows, 704.7 GB parquet) is not the publisher. |

**Discrepancy resolved:** 44,660 h is the **train** split (rounded) and 44,691 h is **all
splits**. Likewise 5,490 speakers is **train** and 5,574 is **all splits** (speaker-disjoint
by MLS design; to verify on download). The registry now stores the exact per-split values.

**New blocker:** `mls_download_provenance_verified = false`. Before GO, the owner downloads
from openslr.org/94 and records the archive sha256 and the licence text on that page.

## 6. MLS subset strategy (not adopted before evidence)

* **Selection:** speaker-balanced, with an hours cap per speaker (solve Σ min(h_i, cap) =
  target).
* **Stratification:**
  * gender balanced (2,742 / 2,748 train);
  * maximise distinct books/chapters (sessions) per speaker;
  * keep the original duration distribution.
* **Accent:** MLS has no accent labels (LibriVox readers, mostly North American). Accent
  diversity comes from VCTK and AMI.
* **Recording conditions:** home recordings; augmentation covers noise, reverb and codec.
* **Transcripts:** all train segments have them.

| Share of train | Hours | Speakers | Storage: mirror opus parquet / 16 kHz FLAC (est.) | Training impact (fixed 720k steps × 16 × 2 s ≈ 6,400 h seen) | GPU / CPU cost difference |
|---|---|---|---|---|---|
| **10 %** | 4,466 | up to all 5,490 (cap ≈ 0.8 h/speaker; exact count needs per-speaker metadata) | 70 GB / 0.31 TB | ≈ 1.4 passes; all speakers; fewer chapters per speaker | teacher units ≈ 2–4.5 A100-h or ≈ 270 CPU-h |
| 25 % | 11,165 | ≤ 5,490 | 176 GB / 0.78 TB | ≈ 0.57 pass; more session diversity | units ≈ 5.6–11 A100-h |
| 50 % | 22,330 | ≤ 5,490 | 352 GB / 1.55 TB | ≈ 0.29 pass | units ≈ 11–22 A100-h |
| 100 % | 44,660 | 5,490 | 705 GB / 3.1 TB | ≈ 0.14 pass | units ≈ 22–45 A100-h |

**Interpretation:**
* At a fixed step budget, the **training** GPU cost is the same for every subset. More data
  adds only preprocessing, storage and download.
* **Proposed default:** 10 %, all speakers, for stages 1–4.
* **Pre-registered rule:** move to 25 % only if the VALID train/valid gap shows data
  limitation in stage 2 or 3.
* This is not adopted until the per-speaker metadata exists (after download).

## 7. Arabic

Plan: `CONSENTED_RECORDING_PLAN.md`.

| | Target |
|---|---|
| Phase A | 500 h / 1,000 speakers (Jordanian 200, Palestinian 150, Syrian 150, Lebanese 100, Egyptian 120, Gulf 100, Iraqi 80, MSA-focused 100) |
| Phase B | 1,000–1,500 h / 1,500–2,500 speakers |

**Consent rules:**
* There are five mandatory scopes: ML training, voice-anonymization R&D, commercial product
  development, model evaluation, derivative model training. Otherwise a speaker is
  `NOT_COMMERCIAL_TRAINING_ELIGIBLE`.
* Per-speaker consent and per-session metadata schemas allow **no PII fields**.
* Device is recorded as a class only.
* Withdrawal and retention are enforced by `datasets/consent.py` in the manifest builder.
* **No** scraping of Telegram, YouTube, news or podcasts.
* **ARABIC_NOT_VERIFIED / ARABIC_NOT_READY.**

## 8. GitHub Actions orchestrator

**Workflow:** `.github/workflows/train-orchestrator.yml`.
* Manual dispatch only; standard free runners only; `contents: read`.
* **`preflight`**: runs `scripts/orchestrate_training.py preflight`, which requires
  `training/readiness.json` verdict GO and all conditions true, a validating dataset
  registry, and optionally `--manifest-dir` verification. **It fails today (NO-GO), so
  nothing downstream runs.**
* **`prepare_training_artifact`**: a deterministic code bundle (tar.gz, mtime 0, sorted, no
  audio, checkpoints or JSONL manifests) plus its sha256. `bundle` itself also refuses to
  run before GO.
* **`launch_external_gpu`**: runs in the protected environment `gpu-launch` (required
  reviewers; provider credentials live only there). It is **not implemented**, because no
  provider is verified, and exits 1.
* **External job contract:**
  1. check the bundle sha256;
  2. `plan-resume --ckpt-dir <storage>` gives the newest **valid** checkpoint (sha256 ==
     sidecar);
  3. `train_from_checkpoint` with atomic checkpoints (`trainers/checkpoint.py`);
  4. upload **only sidecars** (sha256, step, stage).
* **`verify`**: `verify --ckpt-dir … --previous-step N` fails on a corrupt checkpoint or no
  progress, and triggers a resume dispatch.
* **Data steps:** `prepare_dataset` and `validate_manifest` run on the data host
  (`build_manifests.py`, `--verify`, consent gate). GitHub sees only `manifest_index.json`
  hashes.
* **Never** in GitHub: audio, consent records, checkpoints, or secrets outside the protected
  environment. Public-repo artifacts are readable by others.

## 9. Final table

| Resource | GPU | VRAM | Free? | Commercial? | Verified? | Suitable |
|---|---|---|---|---|---|---|
| GitHub GPU larger runner | T4 | 16 GB | **No** ($3.12/h) | yes (paid) | VERIFIED_OFFICIAL | stages 0–6 technically; **not available to this personal account** |
| GitHub standard runners | — | — | yes (public repo) | n/a | VERIFIED_OFFICIAL | orchestrator, tests |
| GitHub Student Pack | — (Codespaces CPU) | — | yes | unknown | VERIFIED_OFFICIAL | no GPU |
| HF ZeroGPU | RTX Pro 6000 | 48/96 GB | 5 min/day | unknown | VERIFIED_OFFICIAL | no (demos) |
| HF Spaces paid | T4…A100 80 GB | 16–80 GB | **No** (A100 $2.50/h) | unknown | VERIFIED_OFFICIAL | price reference |
| Oracle Free Tier | not stated | ? | $300 / 30 days | unknown | VERIFIED_OFFICIAL | GPU eligibility unknown |
| Google Cloud research | GCP | 16–80 GB | ≤ $5k if awarded | unknown | APPLICATION_REQUIRED | research path |
| Oracle / AWS / NVIDIA / RunPod / Vast | various | 16–80 GB | if awarded | unknown | APPLICATION_REQUIRED | if awarded with commercial terms |
| Kaggle / Colab / Lightning / Modal / Azure / Paperspace | T4 / none | 0–16 GB | per search | unknown | SEARCH_ONLY | **not counted** |
| Lambda grant | — | — | ? | unknown | EXPIRED? | no |

| Key | Value |
|---|---|
| **COMMERCIAL_DATA_HOURS** | **44,876 h** (E1 licences): MLS-en 44,691.04, AMI 100, VCTK 44, Speech Commands 29, ClArTTS 12. MLS **download provenance PENDING**. |
| **COMMERCIAL_ARABIC_HOURS** | **12 h** (ClArTTS, 1 speaker, Classical Arabic) |
| **COMMERCIAL_ENGLISH_HOURS** | **44,864 h** |
| **PRIVATE_CONSENT_TARGET_HOURS** | **500 h / 1,000 speakers (Phase A); 1,000–1,500 h / 1,500–2,500 speakers (Phase B)**; 0 recorded |
| **FREE_GPU_VERIFIED_HOURS** | **0** (no source meets all four conditions) |
| **FREE_GPU_UNVERIFIED_HOURS** | ≈ 210 T4-h/month nominal (Kaggle ≈ 130 session-h on 2×T4 + Lightning ≈ 80), plus Modal $30/month. **SEARCH_ONLY; commercial UNKNOWN.** |
| **REQUIRED_GPU_HOURS** | **62–141 A100-h**, ≈ 300–680 T4-h (with 1.3 contingency) |
| **COMPUTE_DEFICIT** | **62–141 A100-h (100 %)**. At the official HF price (A100 80 GB $2.50/h): ≈ **$155–353**, if training in Spaces is permitted (UNKNOWN). At the official GitHub T4 price ($3.12/h): ≈ $940–2,120, and not available to this account. |

## 10. Decision: **NO-GO**

Open conditions (`training/readiness.json`):

| Condition | Status |
|---|---|
| Licence blockers | U1 LibriSpeech for attacker/test; U3 Common Voice |
| `mls_download_provenance_verified` | false |
| `android_runtime_privacy_on_device` | false |
| `gpu_verified_official_free_and_commercial` | false |
| `gpu_budget_approved_by_owner` | false |
| `arabic_scope_decided_by_owner` | false (U4) |

**Fastest legitimate route to GO:**
1. The owner downloads MLS English from openslr.org/94 and records its sha256 and licence.
2. The owner reads Kaggle and Modal terms, or pays for ≈ 62–141 A100-h. HF's official price
   puts that at about $155–353 if allowed.
3. The owner decides U1, U3, U4 and U5.
4. Then set `verdict: GO` in a reviewed commit.

**Not done:**
* no training;
* no GPU spend;
* no download;
* no acceptance criterion changed (privacy/EER/Top-1/WER/ESTOI/MOS/latency 50 ms/RTF/INT8
  ≤ 50 MB/streaming/offline/no cloud).
