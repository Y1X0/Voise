# StreamAnon: evaluation protocol and acceptance criteria

> **Superseded in part by `PRE_TRAINING_TECHNICAL_REVIEW.md` §2–§5:**
> * the acceptance-critical attacker set now also includes **A5 multi-session** and
>   **A6 matched pseudo-speaker**. A6 replaces the earlier "same-session, reported only"
>   condition: an attacker who owns the app can identify and reuse the pool voice;
> * Top-5, attack success at an attacker-calibrated threshold, and threshold transfer are
>   added;
> * "privacy against an evaluator" is separated from "actual resistance";
> * the intelligibility and signal gates are extended (V/UV, F0 correlation, duration,
>   clicks), and PESQ is restricted to reconstruction mode;
> * the protected-voice attribution gate is stricter.
>
> Where the two documents differ, the review document applies.


This protocol is stronger than Phase 3 in four ways:
1. Larger unseen test sets (≥ 40 speakers per language).
2. **Attackers trained on processed speech**: semi-informed, as in the VoicePrivacy Challenge.
3. A cross-device condition.
4. A stricter attribution test.

The existing pipeline is reused unchanged: `scripts/neural_anonymization_eval.py`, with
StreamAnon plugged in as an external system through `training/evaluation/render_onnx.py`.
The render streams frame by frame through the exported (INT8) ONNX model, exactly as the
phone does.

## 1. Test sets (speaker-disjoint from training and from the attacker pool)

| Set | Speakers | Sessions | Style | Purpose |
|---|---|---|---|---|
| Phase-3 corpus (15 speakers; 9 test) | 9 | 3 renders (A/B/C) | mixed | continuity with all earlier numbers |
| LibriSpeech test-clean + test-other | 40 + 33 | chapters | read | comparability with VoicePrivacy; WER with true transcripts |
| VoxCeleb1-O test speakers | 40 | different videos | spontaneous, in the wild | cross-session in real conditions |
| AMI eval meetings | ≥ 16 | meetings | conversational, far-field | spontaneous and noisy |
| Arabic (own recordings + Common Voice ar / FLEURS ar test) | ≥ 40 | ≥ 2 sessions, ≥ 2 devices | see Arabic plan | Arabic; cross-device |
| Cross-device subset | ≥ 20 per language | same speaker on ≥ 3 different phones | read + conversational | enrollment on device X, test on device Y |

## 2. Systems compared (same test sets)

* RAW (no processing)
* DSP Natural / Strong (the app)
* WORLD pseudo-speaker
* kNN-VC and VPC B3 (Phase 3 references)
* StreamAnon-S FP32
* **StreamAnon-S INT8 (the deployable one)**
* StreamAnon-M (if trained)

The INT8 streaming render is the system that passes or fails.

## 3. Attackers (all score processed-enrollment → processed-test unless stated)

| ID | Attacker | Knowledge | Implementation |
|---|---|---|---|
| A0 | Ignorant | enrollment = original speech | all evaluators below |
| A1 | Lazy-informed, pretrained ASVs | processes its enrollment with the system (different session = different pseudo-speaker) | GE2E; **VPC 2024 ECAPA**; **SA-toolkit ResNet-Vox1**; MFCC-statistics (independent, classical); SpeechBrain ECAPA-VoxCeleb and WavLM-Base-Plus-SV **if** their weights become obtainable (currently blocked) |
| A1w | A1 + WCCN back-end | adapts the scoring space on processed TRAIN+VAL speech | existing |
| **A2** | **Semi-informed, fine-tuned** | owns the system; fine-tunes the VPC ECAPA on ≥ 900 attacker-pool speakers processed with random per-session pseudo-speakers | `training/evaluation/semi_informed_attacker.py --mode finetune_vpc` |
| **A3** | **Semi-informed, from scratch** | the same data, trains an ECAPA-TDNN (C=512) from scratch (VPC ASV_eval^anon recipe) | `--mode scratch` |
| A4 | Same-session linkage | enrollment and test from the same session (same pseudo-speaker) | measures within-call linkability. Expected to be high **by design** (one stable voice per call); reported, not gated. |

* **Cross-session:** every attack is run for all session pairs (A→B, A→C, B→C).
* **Cross-device:** enroll on device X and test on device Y, for every pair.

## 4. Metrics

| Metric | How |
|---|---|
| EER, ROC-AUC | `scripts/neural_anonymization_eval.py` |
| Top-1 identification | `scripts/neural_anonymization_eval.py`; report the chance level (1/N) next to it |
| Similarity distributions | same-speaker and different-speaker means, sd, max |
| 95 % CIs | speaker-level bootstrap, 1000 resamples |
| Whisper WER | **absolute** WER with true transcripts (LibriSpeech, FLEURS, own recordings) and **relative** WER vs the original's transcript (all sets) (`scripts/whisper_wer.py`). Arabic uses a multilingual Whisper and reports CER plus normalised WER. |
| ESTOI, DNSMOS (OVRL/SIG/BAK) | existing pipeline. PESQ only where time-aligned, which holds here because the model is length-preserving. |
| Speech dropouts, clipping, duration ratio | existing pipeline |
| Latency | algorithmic, from metadata; mic→ear on the device |
| RTF, CPU, RAM, model size | host + device (`training/android/benchmark_onnx_step.py`; device procedure in the Android plan) |
| Target attribution | `scripts/target_attribution.py`, extended to all training-corpus speakers |
| Human | blind listening (`tools/listening-test/`): intelligibility, naturalness, similarity to the reference, perceived anonymity |

## 5. Acceptance criteria (fixed before training)

Tests use unseen speakers, cross-session, and the INT8 streaming render. "Min" must hold
for the 95 % CI **lower bound** where marked (lb), otherwise for the point estimate.

### Privacy

| Criterion | Min | Target | Rationale |
|---|---|---|---|
| A2/A3 semi-informed EER (strongest attacker) | ≥ 25 % (lb ≥ 20 %) | ≥ 30 % | Best offline VPC 2024 baseline B3: 22–28 % officially. A causal real-time model cannot remove rhythm and contour, so ≥ 35 % against this attacker is not a realistic *acceptance* bar. |
| A1 lazy-informed EER, each pretrained ASV | ≥ 35 % (lb ≥ 25 %) | ≥ 40 % | kNN-VC reached 28–36 % and B3 41–51 % under these exact evaluators |
| A0 ignorant EER | ≥ 40 % | ≥ 45 % | — |
| Top-1, A1 and A2, N ≥ 40 (chance 2.5 %) | ≤ 15 % | ≤ 8 % | RAW ~95–100 %; kNN-VC 26–41 % at N = 9 |
| Consistency | Every evaluator and every session pair meets min; no single evaluator may carry the result | — | Phase-3 rule |
| Anti-impersonation | Mean best-match to any training speaker ≤ p95 of unrelated real voices, **and** < 1 % of utterances above the median same-speaker similarity | — | §6 |

### Utility

| Criterion | Min | Target | Rationale |
|---|---|---|---|
| Absolute WER, LibriSpeech test-clean (Whisper small.en) | ≤ WER(raw) + 3 points | + 1.5 | B3 official: 4.3 vs 1.8 |
| Relative WER, spontaneous/noisy sets | ≤ 20 % | ≤ 12 % | DSP 8–10 %; kNN-VC 34 %; B3 45 % (Phase 3) |
| Arabic CER / normalised WER relative to raw | ≤ 20 % | ≤ 12 % | only once Arabic recordings exist |
| ESTOI | ≥ 0.55 | ≥ 0.65 | DSP Strong 0.54; kNN-VC 0.45 |
| DNSMOS OVRL | ≥ raw − 0.2 | ≥ raw | — |
| Human intelligibility (word transcription by listeners) | ≥ 90 % words | ≥ 95 % | — |
| Human naturalness MOS (1–5) | ≥ 3.0 | ≥ 3.5 | — |
| Speech dropouts | ≤ 1 % of speech frames | — | — |

### Real time / device

| Criterion | Min | Target |
|---|---|---|
| INT8 model size | ≤ 30 MB (hard cap 50) | ≤ 15 MB |
| Added RAM (PSS) | ≤ 150 MB | ≤ 60 MB |
| RTF on 1 big core of the reference mid-range phone | ≤ 0.5 (p99 ≤ 0.8) | ≤ 0.3 |
| Algorithmic latency | ≤ 50 ms | ≤ 40 ms |
| Mic→ear (reference phone, wired headphones) | ≤ 150 ms | ≤ 110 ms |
| 30-min run | 0 deadline-miss bursts; no thermal throttle to fallback | — |

Not met → classification **FAILED** for that model. The utility and privacy blocks
cannot compensate for each other.

## 6. Attribution test (no real person)

1. For every processed test utterance, compute similarity under each A1 evaluator to the
   centroid of **every** speaker of the training corpora (as far as embeddings are
   computable) and of the prior-fitting set.
2. Reference distributions:
   * same-speaker (held-out utterance vs. its own speaker's centroid);
   * unrelated real voices (best match of *original* test speech to the training set).
3. **Pass** = mean best-match ≤ p95(unrelated) and < 1 % of utterances above the median
   same-speaker similarity.
4. Report which training speaker is the most frequent best match. A dominant one means
   the prior has collapsed.

## 7. Statistics

* Speaker-level bootstrap for every EER, AUC and top-1.
* With ≥ 40 test speakers the 95 % CI half-width shrinks to about ±5–7 EER points,
  against ±10–15 at N = 9.
* No test-set tuning: hyper-parameters and the pseudo-pool are chosen on VALID speakers
  only.
* Every failure is reported. Cherry-picking utterances or evaluators is not allowed.

## 8. Commands

```bash
# render + score (pretrained-ASV attackers, quality)
python3 scripts/neural_anonymization_eval.py --corpus <test wav dir> --out eval-streamanon \
  --vpc-asv-dir models/exp/asv_orig --satools-asv-jit models/satools_resnet_v1/final.jit \
  --systems "dsp_strong,psn_world,streamanon=cmd:python3 training/evaluation/render_onnx.py \
     --model build/stream_anon_s.int8.onnx --pool build/pseudo_pool.npy --in {in} --out {out} --seed {seed}"
# semi-informed attackers (GPU), then re-score with the fine-tuned ASV
python3 training/evaluation/semi_informed_attacker.py --manifest data/manifests/attacker.jsonl \
  --anonymizer build/stream_anon_s.int8.onnx --pool build/pseudo_pool.npy --out runs/attacker_ft
python3 scripts/neural_anonymization_eval.py ... --vpc-asv-dir runs/attacker_ft/best
# intelligibility, attribution
python3 scripts/whisper_wer.py --model-dir models/sherpa-onnx-whisper-small.en --systems eval-streamanon/streamanon
python3 scripts/target_attribution.py --out eval-streamanon --systems streamanon --vpc-asv-dir models/exp/asv_orig \
  --satools-asv-jit models/satools_resnet_v1/final.jit
```

## 9. Report

The report is `docs/STREAMING_NEURAL_RESULTS.md`, created only when results exist. It has:
* a table per attacker × evaluator × session pair;
* the utility table, the device table and the attribution table;
* a pass/fail column against §5;
* the final classification.
