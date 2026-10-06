# TRAINING_START_CHECKLIST

`train.py` starts only when **every** box is true. It checks them itself:
`scripts/orchestrate_training.py preflight` and `trainers/train.py` print every missing item
together, each tagged with its category. Run all commands from the repository root, on a
**clean git tree** (real runs refuse a dirty tree).

Categories: **CODE** (a bug in this repository; must always be empty), **DATA**, **GPU**,
**EXTERNAL_POLICY** (owner / legal / provider decision). Today there are **no CODE findings**
(`training/tests/test_consistency.py`); everything open is DATA, GPU or EXTERNAL_POLICY.

Two data scopes:
* `en_only`: `training/configs/data_mix_en.yaml`. **Interim.** Runs are labelled
  ARABIC_NOT_VERIFIED; this does not replace the English + Arabic goal.
* `en_ar`: `training/configs/data_mix_en_ar.yaml`. **The target.** Needs the consented Arabic
  corpus; without it `mix.py` stops with a clear error.

## A. Owner gate (EXTERNAL_POLICY)
- [ ] `training/readiness.json`: `"verdict": "GO"` and every condition of the chosen scope `true`, set in a reviewed commit.
- [ ] Owner decisions U1, U3, U4 (Arabic scope), U5 (`licence_blockers_resolved`, `arabic_scope_decided_by_owner`).
- [ ] On-device runtime privacy test on a real phone (`android_runtime_privacy_on_device`).

## B. GPU
- [ ] (GPU) CUDA GPU visible to PyTorch (`torch.cuda.is_available()`); mixed precision verified on it.
- [ ] Profile matches the card: `a100_40gb`, `a100_80gb` or `t4_16gb`.
- [ ] (EXTERNAL_POLICY) The provider's terms allow training a commercial model, and the budget is approved (≈ 62–141 A100-h, ESTIMATED).

## C. Data provenance (DATA, dataset gate)
- [ ] `data/provenance.json` has `mls_en` with `archive_sha256` + `licence_text_sha256`, written by `datasets/mls.py verify` from the official OpenSLR-94 download.
- [ ] `en_ar` only: consented Arabic recordings with all five commercial scopes, imported by `datasets/arabic_import.py` (`arabic_consented_corpus_available`).

## D. Files `train.py` requires (paths from `training/configs/stream_anon_s.yaml`; `_m` is identical)

| # | Path | Produced by |
|---|---|---|
| 1 | `data/manifests/mix_v1/train.jsonl`, `data/manifests/mix_v1/valid.jsonl` + `data/manifests/mix_v1/manifest_index.json` (must verify) | `datasets/mix.py --config training/configs/data_mix_en.yaml` or `data_mix_en_ar.yaml` (`out: data/manifests/mix_v1`) |
| 2 | `data/features/train/index.jsonl` (every entry with teacher units), `data/features/valid/index.jsonl` | `datasets/feature_cache.py --cache data/features/{train,valid}` (`--units-dir data/teacher/units` for train) |
| 3 | `data/teacher/units/kmeans.npy` (K = `model.n_units` = 500) + per-utterance unit files in `data/teacher/units/` | `scripts/compute_teacher_units.py --out data/teacher/units --k 500` |
| 4 | `models_train/ecapa_train_inhouse.pt` + `.json` (`arch: ecapa`, `role: TRAIN`) and `models_train/resnet34_train_inhouse.pt` + `.json` (`arch: resnet34`, `role: TRAIN`) | `scripts/train_speaker_encoder.py --arch ecapa` / `--arch resnet34 --role TRAIN` |

**Not** required by `train.py` (it never reads them):
* `models_train/whisper-small/`: input of step 3 only (see E).
* `data/manifests/noise.jsonl`, `data/manifests/rir.jsonl`: augmentation is not wired into the
  training loop, so `data.augment.enabled` is `false`; `true` is refused as a CODE finding until
  augmentation is implemented.

## E. Inputs to the data-preparation commands
- [ ] `/data/mls/mls_english/` extracted from the verified archive (official layout: `{train,dev,test}/transcripts.txt`, `segments.txt`, `audio/`). If the audio is `.opus`, the local libsndfile must decode Opus.
- [ ] **Teacher-stage prerequisite:** `models_train/whisper-small/`, a local Hugging Face checkpoint of `openai/whisper-small` (MIT), used only by `compute_teacher_units.py`. Never committed to git.
- [ ] `en_ar` only: `/secure/incoming_v1` (recordings) and `/secure/consents.jsonl` (consent records).
- [ ] Disk: about 70 GB (MLS 10 %, opus) plus decoded/cached audio, teacher units and checkpoints. Plan ≥ 0.5–1 TB SSD.

## F. Commands (exact; checked against the code by `training/tests/test_consistency.py`)

```bash
# preflight (exit 3 while anything is missing; lists code_problems and external_blockers separately)
python3 scripts/orchestrate_training.py preflight --scope en_only
python3 scripts/orchestrate_training.py preflight --scope en_ar

# MLS provenance + manifest
python3 training/datasets/mls.py verify --archive <archive> --licence-text <licence-text-file> --source-url https://www.openslr.org/94/
python3 training/datasets/mls.py manifest --root /data/mls/mls_english --out data/manifests/mls --fraction 0.10

# ENGLISH-ONLY (interim, ARABIC_NOT_VERIFIED)
python3 training/datasets/mix.py --config training/configs/data_mix_en.yaml

# OR ENGLISH + ARABIC (target)
python3 training/datasets/arabic_import.py --incoming /secure/incoming_v1 --consents /secure/consents.jsonl --out /secure/own_recordings_v1 --dataset-version own_ar_v1
python3 training/datasets/mix.py --config training/configs/data_mix_en_ar.yaml

# teacher units + feature cache
python3 training/scripts/compute_teacher_units.py --manifest data/manifests/mix_v1/train.jsonl --teacher models_train/whisper-small --layer 8 --k 500 --out data/teacher/units
python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/train.jsonl --cache data/features/train --units-dir data/teacher/units
python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/valid.jsonl --cache data/features/valid

# speaker encoders: two TRAIN architectures + VALID evaluator
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --arch ecapa --name ecapa_train_inhouse --role TRAIN --out runs/spk/ecapa_train
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --arch resnet34 --name resnet34_train_inhouse --role TRAIN --out runs/spk/resnet34_train
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --arch ecapa --name ecapa_valid_inhouse --role VALID --out runs/spk/ecapa_valid

# training (pick the profile of the card; add --resume after an interruption)
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile a100_40gb --out runs/stream_anon_s_v1
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile a100_80gb --out runs/stream_anon_s_v1
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile t4_16gb --out runs/stream_anon_s_v1
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile a100_40gb --out runs/stream_anon_s_v1 --resume
```

Switching from English-only to English + Arabic rewrites `data/manifests/mix_v1/`; re-run the
teacher-unit, feature-cache, speaker-encoder and training steps afterwards.

## G. Not checked by `train.py`, but required before release
- [ ] VALID evaluators: `ecapa_valid_inhouse` + a VALID ASR (`valid_evaluators_available`).
- [ ] Attacker/test sets (U1).
- [ ] A real Android device for the privacy and RTF tests.
- [ ] Locked acceptance criteria unchanged (`data/acceptance_criteria.json`, sha256-pinned).
- [ ] Arabic: ARABIC_NOT_VERIFIED until a real Arabic evaluation passes.
