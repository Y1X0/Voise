# training/: StreamAnon (design-phase prototype; NOT trained)

Code for a streaming neural speaker anonymizer intended to be trained on a GPU later.
Weights and data never go into git.

* Design: `docs/STREAMING_NEURAL_ANONYMIZER_ARCHITECTURE.md`
* Plans: `docs/STREAMING_NEURAL_{TRAINING,EVALUATION,ANDROID,ARABIC}_PLAN.md`

| Dir | Content |
|---|---|
| `configs/` | `stream_anon_s.yaml` (primary, 6.0 M params), `stream_anon_m.yaml` (11.0 M) |
| `datasets/` | manifest format + validator (`manifest.py`), `example_manifest.jsonl` |
| `models/` | `frontend.py` (causal STFT/mel/prosody, iSTFT-OLA), `stream_anon.py` (model with explicit streaming state), `pseudo_speaker.py` (vetted synthetic-speaker pool) |
| `losses/` | all training losses with formulas |
| `trainers/` | config loader; `train.py` (refuses to run without GPU + assets; `--smoke` = Stage 0); `loop.py` (stages, checkpoint/resume), `abort.py`, `run_info.py` (reproducibility, stage gate), `assets.py`, `smoke.py` |
| `evaluation/` | `render_onnx.py` (phone-identical causal streaming render), `privacy_metrics.py`, `quality_metrics.py`, `attribution.py`, `semi_informed_attacker.py` |
| `export/` | `export_onnx.py` (streaming step, state as I/O) |
| `quantization/` | `quantize_int8.py` |
| `android/` | `benchmark_onnx_step.py` (host compute benchmark); the device procedure is in the Android plan |
| `scripts/` | `compute_teacher_units.py`, `make_pseudo_pool.py`, `measure_training_cost.py` (measured FLOPs/memory per step) |
| `native/` | C binding of the app's unchanged YIN pitch tracker and noise suppressor (runtime-identical features) |
| `smoke/` | Stage-0 smoke manifest (3 CC-BY LibriSpeech excerpts from librosa/data) |
| `tests/` | `test_architecture.py` (26 graph tests), `test_readiness.py` (29: metrics, leakage, gates, abort, telemetry guard, resume) |

```bash
python3 training/tests/test_architecture.py && python3 training/tests/test_readiness.py   # tests (CPU)
python3 training/trainers/train.py --config training/configs/smoke.yaml --smoke --out runs/smoke   # Stage 0 (CPU, ~2 min)
python3 training/datasets/manifest.py validate data/manifests/train.jsonl     # data
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --out runs/s   # GPU only
python3 training/export/export_onnx.py --config training/configs/stream_anon_s.yaml \
    --checkpoint runs/s/qat_int8/generator.pt --out build/stream_anon_s.onnx
python3 training/quantization/quantize_int8.py --in build/stream_anon_s.onnx --out build/stream_anon_s.int8.onnx
python3 training/android/benchmark_onnx_step.py --model build/stream_anon_s.int8.onnx --threads 1 --frames 2
python3 scripts/neural_anonymization_eval.py --vpc-asv-dir models/exp/asv_orig \
    --satools-asv-jit models/satools_resnet_v1/final.jit --systems "streamanon=cmd:python3 \
    training/evaluation/render_onnx.py --model build/stream_anon_s.int8.onnx --pool build/pseudo_pool.npy \
    --in {in} --out {out} --seed {seed}"
```

Untrained exports carry `untrained=1` in their ONNX metadata. The renderer refuses them,
so no untrained model can produce an evaluation "result". They are used only for graph
tests and compute benchmarks.
