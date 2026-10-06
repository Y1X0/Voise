# Free / credit-based GPU options (2026) and a GPU-frugal training plan

**Checked:** 2026-10-06. **Machine-readable:** `training/gpu_profiles.json`.

**Evidence: E2 throughout.** Every provider page is blocked by this environment's egress
proxy (kaggle.com, colab/research.google.com, lightning.ai, modal.com, aws, azure, nvidia,
lambda, runpod, vast.ai, huggingface.co). The figures come from search results that quote
or summarise the official pages. **The owner must re-read each official page before relying
on it.** No figure here is presented as verified.

**Only legitimate use:**
* one account per person;
* no multiple or fake accounts, no VPN or region games, no referral or credit abuse, and
  no workarounds of session limits or terms of service.

Colab's FAQ explicitly disallows "using multiple accounts to work around access or resource
usage restrictions", and Kaggle's terms allow one account per person.

## 1. Options

| Provider | GPU | VRAM | Free amount | Session | Persistent storage | Background jobs | Multi-account ban | Commercial ML training allowed? | Requirements | A100-eq h/month (guaranteed) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kaggle** | 2× T4 or 1× P100 | 16 each | ≈ 30 GPU-h/week | ≈ 9–12 h | dataset/output versions | yes (commit runs) | **yes** | **NOT_VERIFIED** | phone-verified account | **≈ 49** (2×T4 DDP) |
| **Lightning AI** (free) | T4-class | 16 | 15 credits/month (≈ 80 GPU-h) | restart ≈ 4 h | yes | limited | NOT_VERIFIED | NOT_VERIFIED | phone verification | **≈ 17** |
| **Modal** (Starter) | T4…H100 | 16–80 | $30/month credits | configurable | Volumes | yes | NOT_VERIFIED | likely (paid cloud), NOT_VERIFIED | account | **≈ 8** (prices NOT_VERIFIED) |
| Google Colab (free) | T4 when available | 16 | unpublished, dynamic | ≤ 12 h; idle cut | Drive only | no | **yes** | NOT_VERIFIED | Google account | 0 (not guaranteed) |
| Paperspace Gradient (free) | M4000 | 8 | no guaranteed hours | 6 h | 5 GB | no | NOT_VERIFIED | NOT_VERIFIED | account | 0 (8 GB unusable) |
| HF ZeroGPU | shared | — | ≈ 5 min/day | per request | — | no | — | not a training service | account | 0 |
| Azure for Students | **no GPU quota** | — | $100 / 12 months | — | — | — | — | — | student | 0 |
| Google Cloud research credits | any GCP GPU | 16–80 | ≈ $1k (PhD) / ≈ $5k (faculty) | — | yes | yes | — | **NO**: "may not be used for commercial purposes" | academic; eligible country; proposal | application |
| Google TRC | **TPU** | — | free TPU allocation | allocation | GCS (paid) | yes | — | research; publication expected | application | application (needs a PyTorch/XLA port) |
| AWS Cloud Credits for Research | any EC2 GPU | 16–80 | by proposal | — | yes | yes | — | NOT_VERIFIED (research award) | academic; quarterly deadlines | application |
| NVIDIA Academic Grant | H100/A100 hours | 80 | up to ≈ 30,000 H100-h | ≤ 8 concurrent | — | yes | one award/year | NOT_VERIFIED | accredited institution; must use NVIDIA software stack | application |
| Lambda Research Grant | Lambda cloud | 24–80 | up to $5k | — | yes | yes | — | NOT_VERIFIED | researchers | application (status since Dec 2024 unverified) |
| RunPod research / startup | A100/H100/4090 | 24–80 | research ≤ $25k; startup $1k | — | network volumes | yes | NOT_VERIFIED | startup track: likely, NOT_VERIFIED | application | application |
| Vast.ai Startup Program | marketplace | 24–80 | ≤ $2.5k | — | instance disk | yes | NOT_VERIFIED | NOT_VERIFIED | startup | application (third-party hosts: data-security review) |
| Oracle for Research | OCI GPUs | 16–80 | project awards | — | yes | yes | — | research | academic proposal | application |
| University / national HPC | A100/H100 | 40–80 | allocation | queue | yes | batch | — | usually research only | affiliation | application |

### Guaranteed free GPU found (no application)

| | Value |
|---|---|
| Nominal | **≈ 74 A100-equivalent hours per month** (Kaggle 49 + Lightning 17 + Modal 8) |
| Effective | **≈ 35–45 A100-eq h/month** after interruptions, fp16-only GPUs, 16 GB batch limits and I/O |
| Commercially usable | **0 VERIFIED.** No free tier's terms were read to confirm that training a commercial product's model is allowed (all NOT_VERIFIED). The academic credits explicitly or probably exclude commercial use. |

**A100-equivalence** uses peak tensor throughput (T4 65 vs A100 312 TFLOP/s fp16/bf16,
recalled, NOT VERIFIED).
* T4 and P100 have **no bf16**, so stages 2–4 run in fp16 with loss scaling.
* GAN fp16 stability is a risk, to be checked in Stage 2.
* At batch 8 + gradient accumulation 2, a run fits 16 GB (`TRAINING_COMPUTE_ESTIMATE.md` §4).

## 2. Training plan that minimises GPU use

Total FLOPs come from the measured per-step costs in `TRAINING_COMPUTE_ESTIMATE.md`.

* Stage-3 step ≈ 6.7 TFLOP. The content-output loss now uses the frozen stage-1 encoder,
  not a HuBERT teacher.
* A100 effective throughput is 10–20 % MFU.
* No 250-hour continuous run is assumed. Every stage is split into resumable chunks that
  fit a 9–12 h session.

| Stage | What | VRAM | A100-h | T4-h (≈ ×4.8) | Checkpoint | Resume | Output | GO / NO-GO gate |
|---|---|---|---|---|---|---|---|---|
| **0** Small smoke | mechanics on 45 s CC-BY | CPU | 0 | 0 | 0.27 GB | not needed | S1–S8 report | **passing (CI)** |
| **1** Teacher units + content distillation | Whisper-small units (offline, idempotent per utterance) + encoder/VQ/unit/CTC heads, 200k steps | ≈ 4 GB | 3–7 | 15–35 (units also on CPU) | ≈ 0.1 GB | every 2,000 steps; unit files skip-if-exists | units + stage-1 encoder | stage-1 exit criteria on VALID (`TRAINING_READINESS_GATE.md` §1) |
| **2** Small-scale training | 30k reconstruction + 20k anonymisation steps on a 100–300 h subset | 14 GB bf16 / 16 GB fp16 at batch 8 + accum 2 | 1.1–2.3 | 5–11 | 0.27 GB | every 1,000 steps (≈ 10–20 min on T4) | first anonymised VALID audio; real-data end-to-end check; fp16 stability check | losses fall, no abort, VALID WER/EER computed. **No privacy claim.** |
| **3** Ablations | ≤ 8 pre-registered runs × 20k steps: look-ahead 2/4, window 320/640, teacher layer, loss weights | as stage 2 | 4.8–9.5 | 23–46 | 0.27 GB | every 1,000 steps; runs sequential | decisions recorded on VALID | choices fixed **before** stage 4 |
| **4** Full training | 300k reconstruction + 200k anonymisation steps | as stage 2 | 11.4–22.9 | 55–110 | 0.27 GB; keep last 3 + best + exits | every 1,000 steps | trained FP32 generator | stage-3 exit criteria on VALID, **unchanged** |
| **5** QAT / INT8 | 20k steps + ExecuTorch export | ≈ 10 GB | 0.4–0.7 | 2–4 | 0.27 GB | every 1,000 steps | INT8 `.pte` (S ≈ 7.6 MB) | INT8 vs FP32: VQ agreement ≥ 98 %, ΔWER ≤ 2 pts, \|ΔEER\| ≤ 3 pts |
| **6** Final locked evaluation | attackers (A2/A3) + `final_eval.py` (one-shot lock) + protocol S1–S5 | 16 GB | 4–10 | 20–50 | 0.1 GB | attacker training resumable; final evaluation one-shot | `final_eval.json`, protocol reports | **all acceptance criteria (§4), unchanged** |
| Support | in-house speaker encoders (2 TRAIN + 1 VALID), validation overhead | 16 GB | 23–56 | 110–270 | 0.1 GB | every 2,000 steps | frozen encoders | encoder VALID EER recorded |
| **Total** | | | **≈ 48–108** (× 1.3 contingency = **62–141**) | **≈ 230–520** | | | | |

This is about half of the earlier programme figure (120–250 A100-h). The difference comes
from:
* one full run plus pre-registered short ablations, instead of 2 full runs plus 4
  one-third-length ablations;
* a cheaper stage-3 content loss.

If the first full run misses its gates, a second full run costs about +12–23 A100-h more.
**No acceptance criterion was changed to reach this budget.**

### Resumability (implemented and tested)

| Requirement | Where |
|---|---|
| Checkpoint every N steps | `Trainer.run_stage(ckpt_every=…)`; numbered `step_<n>` tags rotate (keep 3) |
| Atomic checkpoint write | `trainers/checkpoint.py`: temp file → fsync → `os.replace` → fsync dir; a sha256 sidecar is written afterwards. Tested: an interrupted write leaves the previous file, and no temp files remain. |
| Resume after disconnect | `checkpoint.latest_valid(dir)` returns the newest checkpoint whose sha256 matches its sidecar, skipping corrupt or partial ones (tested). `Trainer.load` restores model, optimisers, step, stage, history, **torch/cuda RNG**. |
| Deterministic seed | batches are a pure function of (seed, step) (`segments.py`). Resume is bit-exact (smoke S4). |
| Dataset manifest hash | `manifest_index.json` sha256; `build_manifests.py --verify` in preflight |
| Code commit hash | `run_info.json` git SHA + dirty flag (real runs refuse a dirty tree) |
| Environment lock | `pip freeze` saved + sha256 |
| GPU metadata | `run_info.env.gpu`, CUDA version |
| Training config hash | `config_sha256` |

**Free-tier specifics:**
* Write checkpoints to the persistent location: Kaggle output/dataset version, a Lightning
  Studio disk, or a Modal Volume.
* Size each session's work to finish one checkpoint interval before the session limit.
* Never leave training data on third-party marketplace hosts without a security review.

## 3. GPU balance

| | A100-eq hours |
|---|---|
| Required (with contingency) | **62–141** |
| Guaranteed free found, nominal / effective | ≈ 74 / ≈ 35–45 **per month** |
| Calendar time on free tiers alone | ≈ **2–4 months**, *if* their terms allow commercial ML training (NOT_VERIFIED) |
| Free GPU-hours VERIFIED usable for a commercial model | **0** |
| **Remaining deficit**, strict reading | **62–141 A100-h** (all of it) until the owner confirms the free tiers' terms, or an award/credit with commercial-use permission is obtained |
| Remaining deficit, if Kaggle + Lightning + Modal terms permit commercial training | **0 in hours**, ≈ 2–4 months of calendar time, with fp16/16 GB risk |

**Fastest legitimate routes:**
1. The owner reads the Kaggle / Lightning / Modal terms on their official pages (minutes).
2. Apply for the RunPod startup or research credits, or Vast.ai startup credits; these are
   commercial programmes.
3. If an academic affiliation exists: the NVIDIA Academic Grant (covers everything if
   awarded). The resulting model would then be research-only unless the award terms allow
   commercial use.

## 4. Acceptance criteria: unchanged

These were **not** changed for lack of GPU:

| Area | Criterion |
|---|---|
| Privacy | informed attacker EER ≥ 25 % minimum (target ≥ 30 %); strong pretrained attacker target ≥ 35–40 %; Top-1 ≤ 15 % (target ≤ 8 %) with ≥ 40 speakers |
| Intelligibility | LibriSpeech WER ≤ original + 3 absolute points; spontaneous relative WER ≤ 20 % (target 12 %); ESTOI ≥ 0.55; human MOS ≥ 3.0 |
| Real time | algorithmic latency ≤ 50 ms; causal streaming; no full-utterance dependency; RTF < 0.5 on a representative Android big core |
| Model | INT8 model ≤ 50 MB (the internal export gate stays at ≤ 30 MB, stricter) |
| Network | no cloud; no network permission or runtime network dependency |
| Arabic | **ARABIC_NOT_VERIFIED** until an actual Arabic evaluation passes |
