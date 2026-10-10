# Free path for Arabic voice anonymization: candidate decision (2026-10)

Scope: personal/private use, zero cost, local Android, streaming preferred. Research only:
nothing was trained, the DSP/Android/Termux code is unchanged, and `data/acceptance_criteria.json`
is unchanged (sha256 prefix 58a893c2). No model below is claimed to anonymize until it passes an
independent speaker-verification test.

## 1. Closed: StreamVoiceAnon + torch.compile

`experiments/streamvoiceanon_t4/compile_benchmark_report.json` (Kaggle T4, torch 2.11):

| | eager | torch.compile (repo settings) |
|---|---|---|
| Stream chunk p50 (46.4 ms deadline) | 125.5 ms | 143.7 ms |
| Stream RTF | 2.72 | 3.11 |
| Offline RTF | 1.34 | 1.84 |
| Compile warm-up | – | 607 s |

torch.compile did not improve speed on T4. The causes are recorded in the report: no native
bfloat16 on T4, CUDA graphs skipped (in-place op), and complex STFT ops. **This experiment is not
repeated with the same settings.** StreamVoiceAnon remains an offline Arabic-intelligibility
reference only.

## 2. Repository: reuse vs keep unchanged

**Keep unchanged**
- `dsp/`, `app/`, `termux/`: the shipped DSP product.
- `data/acceptance_criteria.json`: locked.
- `training/readiness.json`: NO-GO.

**Reuse**
- Privacy evaluation (`training/evaluation/`): informed/semi-informed attacker, bootstrap EER,
  acceptance check, and the 4-evaluator protocol of `docs/NEURAL_MODEL_EVALUATION.md`.
- kNN-VC and VoicePrivacy B3 setups: offline teachers toward synthetic pseudo-speakers, already
  measured as strong against the informed attacker.
- `training/export/` (ONNX, ExecuTorch) and `training/android/benchmark_onnx_step.py` for any
  small causal model.
- Kaggle tooling and the T4 measurements.
- `data/dataset_registry.json`, and the Arabic source audit (Omnilingual, SVQ: CC BY 4.0).

## 3. Candidates (max 3)

Type matters:
- **TTS** (e.g. MMS-TTS) makes speech from text and is not an anonymizer; it was only used to
  generate synthetic test input.
- **Voice conversion (VC)** maps speech to a target voice.
- **Voice anonymization (VA)** is VC (or DSP) evaluated against speaker verification.

| | 1. LLVC (KoeAI) | 2. kNN-VC / VPC B3 (in repo) | 3. StreamAnon-S (ours) |
|---|---|---|---|
| Type | causal any-to-one VC | offline VC / VA | streaming VA (designed) |
| Code / weights (verified) | github.com/KoeAI/LLVC (MIT, 2023-11); huggingface.co/KoeAI/llvc (MIT): `G_500000.pth` 39.5 MB | official releases, already in `models/` | `training/` (85 files); no weights |
| Pretrained checkpoint | yes, one target voice: `f_8312`, a LibriSpeech speaker rendered by RVC (+12 semitones) | yes | no (training NO-GO: data) |
| Inference (verified in code) | waveform in/out, 16 kHz, chunk 13×16 = 208 samples (13 ms); `infer.py -s` streaming; needs only `model.py` + `cached_convnet.py` + checkpoint | utterance-level (WavLM-Large ~300M / STTTS) | causal, 40 ms, ~6M params |
| Streaming | yes (cached causal conv + causal transformer decoder) | no | yes (design) |
| Arabic | NOT VERIFIED (trained on English LibriSpeech) | NOT VERIFIED | needs Arabic data + training |
| Measured privacy | none | strong vs informed attacker (English), intelligibility loss | none |
| Kaggle free | yes (tiny) | yes | training only (T4 measured) |
| Android | plausible: standard ops (Conv1d, ConvTranspose1d, LayerNorm, causal TransformerDecoder with explicit caches); not exported yet | no (too large / slow) | designed for it (int8 ≤ 30 MB) |
| Licence (personal use) | MIT code + weights; target voice from LibriSpeech (CC BY 4.0) | MIT / VPC licences | own code; data licences per registry |

Not shortlisted:
- StreamVoiceAnon: measured, not real-time, about 1.5 GB.
- Seed-VC and RVC: GPU-class, reference/target-voice cloning, not phone-realistic.
- StreamVC: no public weights.

## 4. Decision

**Recommendation: LLVC (candidate 1)**, tested as-is first. It is the only one with released
weights that is causal, small (39.5 MB) and built from standard ops.

Known caveats:
- Its released target is a real LibriSpeech reader. For deployment, the target should be
  retrained toward a synthetic pseudo-speaker, using the repo's kNN-VC/B3 teachers to make the
  training pairs.
- Arabic intelligibility and privacy are unmeasured.

Kill criteria for the first test: if Arabic CER rises a lot, or the independent ASV EER fails,
LLVC is dropped and StreamAnon-S stays the long-term path.

## 5. Smallest next experiment (not run: needs licensed Arabic real speech)

The repo has no licensed real Arabic speech: `eval-corpus` is English (AMI CC BY 4.0, plus
`eval-audio`). Real multi-speaker audio is required, because a synthetic voice cannot test speaker
identity.

**Proposed source:** a small SVQ Arabic subset (google/svq, CC BY 4.0, `speaker_id`, 4 locales):
about 12 speakers × 4 clips (~50 short files, a few MB), pulled inside Kaggle.

**Run, on Kaggle T4 + CPU:**
1. LLVC `G_500000.pth`, offline and streaming (`-s`, chunk factor 1 and 2).
2. Measure:
   - RTF on GPU and on 1 CPU thread, plus per-chunk deadline analysis;
   - Whisper-small Arabic CER, original vs output;
   - privacy with two independent evaluators that are not part of LLVC (SpeechBrain ECAPA
     VoxCeleb, Apache-2.0; WavLM-base-plus-sv):
     - ignorant attacker: original enrolment vs anonymized trial;
     - lazy-informed attacker: anonymized vs anonymized;
     - EER with bootstrap CI.
3. No training, no other data.

CER/WER alone is never treated as evidence of anonymization.
