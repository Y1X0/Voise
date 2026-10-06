# Neural model offline evaluation (Phase 2)

**Classification: D — MODEL_UNAVAILABLE / INCONCLUSIVE.**
No neural anonymization model was evaluated. Not a single admissible model's weights
could be obtained in this environment (`NEURAL_MODEL_ACQUISITION.md`). The status of
every model is therefore **MODEL_UNAVAILABLE**, and nothing in this phase is FAILED or
OFFLINE_VALIDATED.

## Status summary

| Item | Status | Evidence |
|---|---|---|
| kNN-VC → synthetic pseudo-speaker | MODEL_UNAVAILABLE (MODEL_ARTIFACTS_REQUIRED) | GitHub release assets: 403 |
| VoicePrivacy 2024 B3 (STTTS + WGAN) | MODEL_UNAVAILABLE (MODEL_ARTIFACTS_REQUIRED) | GitHub release assets: 403 |
| LLVC | EXCLUDED_BY_POLICY | released checkpoint targets a real LibriSpeech speaker |
| VoicePrivacy B5/B6 | EXCLUDED_BY_POLICY | targets real LibriTTS training speakers |
| Streaming anonymizers (Quamer 2024, DarkStream, Stream-Voice-Anon, TVTSyn), StreamVC | MODEL_UNAVAILABLE | no public weights |
| Modern evaluator ECAPA-TDNN | **NOT_VERIFIED** | huggingface.co not accessible |
| Modern evaluator WavLM-SV | **NOT_VERIFIED** | huggingface.co not accessible |
| Arabic | **ARABIC_NOT_VERIFIED** | no Arabic corpus |
| Human listening test | not run | only required for a promising model; there is none |
| Android / ONNX / TFLite | not started | forbidden in this phase, and the offline gate was not passed |

The earlier non-neural results are unchanged:

* DSP presets and `psn_world` (`NEURAL_ANONYMIZATION_RESULTS.md`, `docs/results/neural_anonymization_tables.md`);
* these stay judged by GE2E + MFCC-stats only;
* that is **not sufficient**, per the phase rule, until ECAPA/WavLM are run.

## What was done in this phase (and verified)

1. **Optional ECAPA and WavLM evaluators added to the existing pipeline.** This was not
   a new evaluator: two classes and two flags were added to
   `scripts/neural_anonymization_eval.py`:
   * `--ecapa-dir` → evaluator `ecapa` (SpeechBrain `EncoderClassifier`);
   * `--wavlm-sv-dir` → evaluator `wavlm_sv` (transformers `WavLMForXVector`).

   Both load only from local files. With `HF_HUB_OFFLINE=1`, a missing file is an error,
   never a silent download. Everything downstream is unchanged for every evaluator:
   * the same trials and attacks (ignorant, lazy same, lazy cross A→B/A→C/B→C, WCCN);
   * the same speaker-level bootstrap CIs and top-1;
   * the same split (`docs/results/neural_splits.json`, which was not modified) and the
     same seeds.
2. **Plumbing check with random weights.** The test only shows the code path works:
   * Checkpoints were built locally from the official configs (`hyperparams.yaml` read
     through the Hub connector; WavLM reduced to 2 layers).
   * The full pipeline ran end-to-end on `dsp_natural` with all four evaluators, exit code 0.
   * Embedding sizes were ECAPA 192 and WavLM 512, unit-norm.
   * The resulting numbers come from untrained networks, so they mean nothing; they were
     discarded and are not reported anywhere.
3. **kNN-VC adapter** `scripts/model_adapters/knnvc_pseudo.py`:
   * The matching set is the `psn_world` renders of TRAIN speakers for the same session,
     i.e. a synthetic pseudo-speaker, with no real target person and no test speaker.
   * **NOT EXECUTED**: running third-party model code was refused by the session policy.

## Evaluation plan once the files are supplied (fixed now, before seeing any result)

The protocol is the existing one, with no change to the acceptance criteria:

* 9 unseen test speakers; TRAIN/VAL used only for `psn_world` statistics and the WCCN
  attacker;
* sessions A/B/C = seeds 101/202/303;
* 1000 speaker-level bootstrap resamples;
* no parameter search on test speakers.

The only fixed model setting is kNN-VC `topk=4`, the authors' default; it is not tuned.

Order of runs:

1. Re-run the DSP presets and `psn_world` with `ecapa` + `wavlm_sv` (§3.1–3.2 of the
   acquisition doc). This tells whether the GE2E / MFCC conclusions of the previous
   phase hold under modern evaluators.
2. Run kNN-VC → pseudo-speaker (§3.3).
3. Run VPC B3 (§3.4) if its environment can be built.

A model is **OFFLINE_VALIDATED** only if it meets all the phase acceptance criteria:

* lower linkability under every evaluator (GE2E, MFCC-stats, ECAPA, WavLM), not only one;
* the improvement holds with bootstrap CIs that exclude the DSP/psn_world baseline;
* top-1 identification drops;
* a low VAD/embedding failure rate (failures are counted, never dropped);
* quality does not collapse (ESTOI, relative WER, DNSMOS against the frozen DSP);
* the result repeats across sessions A/B/C and the WCCN attacker;
* no tuning on test speakers.

Any failure is recorded as FAILED. No utterances are cherry-picked, and the evaluator is
not modified after results are seen.

## Final decision

**D — MODEL_UNAVAILABLE / INCONCLUSIVE.** The project has no evidence for or against
neural anonymization under the informed-attacker threat model. The previous finding
stands (`NEURAL_ANONYMIZATION_NOT_VALIDATED`). Next step: provide the files listed in
`NEURAL_MODEL_ACQUISITION.md` §3, in order of priority:

1. ECAPA + WavLM (≈ 494 MB);
2. kNN-VC (≈ 1.25 GB);
3. optionally B3 (≈ 1.1 GB, GPL-3.0).
