# Kaggle T4 smoke (GPU regression gate)

Three bugs reached a T4 before anyone saw them, because CPU tests cannot catch a tensor on the wrong
device. Each one only failed on CUDA:

| commit | bug (CUDA only) | regression test (`tests/test_kaggle.py`) |
|---|---|---|
| 8e01a50 | `_init_codebook` mixed CPU and CUDA tensors | `CodebookInit` |
| d108a58 | pseudo-speaker noise in `step_anon` / `step_qat` created on the CPU | `PseudoSpeakerNoiseDevice` |
| bb66a0e | `Trainer.load` mapped the saved RNG states to `cuda:0`; `torch.set_rng_state` refused them, so `--resume` failed (smoke step 8) | `ResumeRngOnDevice` |

This folder runs those GPU tests and the full session smoke (`kaggle.py smoke`, docs/KAGGLE.md) on a
Kaggle T4 for one commit, from a machine without a GPU. **Run it after every change to the trainer,
checkpoints or device handling, and before any training session.** A new CUDA-only bug gets a test
class here and its name added to `GPU_TESTS` in `run.py`.

## Run

```bash
pip install kaggle                      # KAGGLE_USERNAME / KAGGLE_KEY set in the environment
git push                                # the kernel clones the commit from GitHub
python3 training/configs/kaggle/kernel_t4_smoke/smoke.py --ref $(git rev-parse HEAD)
```

`smoke.py` pushes `run.py` with the commit filled in (kernel `<user>/voise-t4-smoke-<sha7>`, private,
T4, internet on), waits for it, reads the log through the Kaggle API and writes
`runs/kaggle_t4_smoke/<sha7>/kernel.log` + `session_smoke_report.json`. Exit 0 only when the GPU tests
pass and the report has `passed: true` and `kaggle_validated: true`. It takes about 5 minutes.

What the kernel does: `nvidia-smi`; clone + checkout of the exact commit; `speechbrain==1.1.1`
(`--no-deps`) + `hyperpyyaml` + `ruamel.yaml`; `path_map.example.json` to `/kaggle/working/`;
`test_kaggle.py CodebookInit PseudoSpeakerNoiseDevice ResumeRngOnDevice -v`; `kaggle.py smoke` with
`PYTHONFAULTHANDLER=1 timeout -s SIGABRT 2400`.

No datasets are attached, so the smoke uses synthetic batches and randomly initialised speaker
encoders (`2_dataset_paths: MISSING` is expected). It proves the code on the GPU, not the data.

## Recorded runs (`reports/`)

| commit | result | notes |
|---|---|---|
| d108a58 | FAIL at `8_process_restart` | RNG state on `cuda:0` at resume (fixed in bb66a0e) |
| c8ffc51 | PASS, `kaggle_validated: true` | all 10 checks; resume bit-exact (max abs diff 0.0); peak 7.89 GB reserved of 15.64 GB (anonymization, batch 16, fp16) |
