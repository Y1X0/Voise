# TRAINING_START_CHECKLIST

`train.py` starts only when **every** box is true. It checks them itself:
`scripts/orchestrate_training.py preflight` and `trainers/train.py preflight` together print every
missing item. Run all commands from the repository root, on a **clean git tree** (real runs refuse a dirty
tree).

## A. Owner gate
- [ ] `training/readiness.json`: `"verdict": "GO"` and **every** `conditions` value `true`, set in a reviewed commit.

## B. GPU
- [ ] CUDA GPU visible to PyTorch (`torch.cuda.is_available()`).
- [ ] Profile matches the card: `a100_40gb`, `a100_80gb` or `t4_16gb`.
- [ ] The provider's terms allow training a commercial model, and the budget is approved (≈ 62–141 A100-h, ESTIMATED).

## C. Data provenance (dataset gate)
- [ ] `data/provenance.json` has `mls_en` with `archive_sha256` + `licence_text_sha256`, written by `datasets/mls.py verify` from the official OpenSLR-94 download.
- [ ] The same for every other training corpus in the mix (e.g. `vctk`, `ami`) via `mls.py verify --corpus … --source-url …`, or remove that corpus from `training/configs/data_mix.yaml`.

## D. Files `train.py` requires (paths from `training/configs/stream_anon_s.yaml`)

| # | Path | Produced by |
|---|---|---|
| 1 | `data/manifests/train.jsonl`, `data/manifests/valid.jsonl` + `data/manifests/manifest_index.json` (must pass `--verify`) | `mix.py`. **Config mismatch:** `mix.py` writes to `data/manifests/mix_v1/`, so set `data.train_manifest` / `data.valid_manifest` to `data/manifests/mix_v1/{train,valid}.jsonl` (or set the mix `out:` to `data/manifests`). |
| 2 | `data/features/train/index.jsonl`, `data/features/valid/index.jsonl` | `datasets/feature_cache.py` |
| 3 | `data/teacher/units_k500.npy` | **Config mismatch:** `compute_teacher_units.py` writes `data/teacher/units/kmeans.npy` + per-utterance files; set `data.teacher.units` to `data/teacher/units/kmeans.npy`. |
| 4 | `models_train/ecapa_train_inhouse.pt` **and** `models_train/resnet34_train_inhouse.pt` (TorchScript) | `scripts/train_speaker_encoder.py --role TRAIN`. The script trains ECAPA for both names. Either train a second TRAIN encoder under that name, or list only `ecapa_train_inhouse` in `data.speaker_encoders_train`. |
| 5 | `data/manifests/noise.jsonl`, `data/manifests/rir.jsonl` | **No builder exists.** Provide them (DNS Freesound-CC0 noise / DNS-redistributed RIRs, provenance recorded) or remove `data.augment.noise_manifest` / `rir_manifest` from the config. Note: augmentation is **not wired into the training loop**; only preflight checks these files. |

## E. Inputs to the data-preparation commands
- [ ] `/data/mls/mls_english/` extracted from the verified archive (official layout: `{train,dev,test}/transcripts.txt`, `segments.txt`, `audio/`). If the audio is `.opus`, the local libsndfile must decode Opus.
- [ ] `models_train/whisper-small/`: a local Hugging Face checkpoint of `openai/whisper-small` (MIT).
- [ ] `training/configs/data_mix.yaml` lists **only sources that exist**. For English-only, set `language_weights: {en: 1.0, ar: 0.0}` and remove `own_ar_v1`, otherwise `mix.py` fails.
- [ ] Disk: about 70 GB (MLS 10 %, opus) plus decoded/cached audio, teacher units and checkpoints. Plan ≥ 0.5–1 TB SSD.

## F. Not checked by `train.py`, but required before release
- [ ] VALID evaluators: `ecapa_valid_inhouse` + a VALID ASR.
- [ ] Attacker/test sets (U1).
- [ ] A real Android device for the privacy and RTF tests.
- [ ] Locked acceptance criteria unchanged (`data/acceptance_criteria.json`, sha256-pinned).
