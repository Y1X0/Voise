# Kaggle T4 session smoke: runtime benchmark

**What this is:** a runtime benchmark of the training code on a free Kaggle Tesla T4
(memory fit, numerical stability, checkpoint/resume). It says nothing about anonymization quality,
and no acceptance criterion (`data/acceptance_criteria.json`) is measured or claimed here.

**Input:** synthetic, shape-identical batches and randomly initialised speaker encoders of the real
architectures (ECAPA-TDNN, ResNet-34), held in memory only. **No real training data was used and
real-data training has NOT started.** Memory with the trained encoders and real audio has not been
measured.

## Run

| Item | Value |
|---|---|
| Commit | `bb66a0e` (`bb66a0e36e0135a687ca1848fa42fe37b80ee43b`) |
| Command | `python3 training/trainers/kaggle.py smoke --config training/configs/stream_anon_s.yaml --gpu-profile t4_16gb --out /kaggle/working/session_smoke --path-map /kaggle/working/path_map.json` (see `docs/KAGGLE.md`) |
| GPU | NVIDIA Tesla T4 (Kaggle notebook, T4 ×2 attached; only `cuda:0` used) |
| PyTorch-visible VRAM | 15.64 GB |
| Batch size | 16 (profile `t4_16gb`, unchanged) |
| Precision | fp16 (autocast + GradScaler) |
| Report | `session_smoke_report.json`: `passed: true`, `kaggle_validated: true` |

## Measured GPU memory per stage (MEASURED)

Peak allocated / peak reserved over the smoke steps of each stage at batch 16, fp16:

| Stage | Peak allocated | Peak reserved | Status |
|---|---|---|---|
| content_distillation | 0.40 GB | 0.44 GB | FITS |
| reconstruction | 4.60 GB | 5.41 GB | FITS |
| anonymization | 7.31 GB | 7.89 GB | FITS |
| qat_int8 | 2.57 GB | 7.89 GB | FITS |

All losses were finite, and no abort event (NaN/Inf, collapse) or CUDA OOM occurred.
The qat_int8 reserved figure includes memory still cached from the preceding stage.

| Other measurement | Value |
|---|---|
| System RAM peak (smoke process) | 3.15 GB |
| CUDA regression tests (`CodebookInit`, `PseudoSpeakerNoiseDevice`, `ResumeRngOnDevice`) | 6/6 OK |
| Resume in a new process vs uninterrupted run | bit-exact, max abs difference 0.0; step counters, schedulers, GradScaler and RNG states equal |

## Note on determinism

PyTorch reported one operation without a deterministic CUDA implementation:
`reflection_pad1d_backward_out_cuda`. In this short run, two uninterrupted runs and the resumed run
were identical (difference 0.0). Over long training, bit-exact equality between separate GPU runs is
therefore not guaranteed. Resume itself restores the full training state either way.

## Conclusion

* T4 environment validated (`kaggle_validated: true`).
* Batch 16 + fp16 fits on the T4 in all four stages (peak 7.89 GB reserved of 15.64 GB).
* No OOM and no NaN observed.
* Resume state restoration validated on the GPU (new process, bit-exact in this run).
* Real-data training has NOT started; MLS download provenance is still PENDING and
  `training/readiness.json` is still NO-GO.
