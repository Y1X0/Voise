# Model-path decision (2026-10-10): **TRAIN LLVC selected — NOT STARTED, blocked**

- No experiment was run for this decision. It uses existing evidence only.
- The original verdict of the 45-speaker evaluation stays **INCONCLUSIVE**.
- The acceptance criteria are unchanged.

## Evidence used (all already committed)

| Item | Measured value | Source |
|---|---|---|
| Pretrained LLVC vs DSP-strong, lazy-informed EER gain | ECAPA +0.087, CI [+0.028, +0.139]; WavLM +0.091, CI [−0.003, +0.173] → G1 not established | `../llvc_vs_dsp_48spk/REPORT.md`, `secondary/uncertainty.json` |
| LLVC intelligibility (relative WER increase vs original) | whisper +1.68, MMS +0.65 (locked target ≤ 0.20). Worse than DSP-strong for 42/44 and 43/43 speakers (reference: transcript_truth) | `REPORT.md`, `secondary/reference_audit.json` |
| ONNX fp32 vs PyTorch | not materially different: largest \|ΔEER\| 0.0001, ΔCER 0.000, minimum SNR 85.5 dB; 1 thread p95 12.7 ms per 26 ms chunk, RTF 0.446 (server CPU) | `secondary/onnx_validation_results.json` |
| Arabic multi-speaker data with speaker labels (CC BY 4.0) | Omnilingual: 184 config-scoped speaker keys, 225.8 h, 19 varieties; splits speaker-disjoint within a config; IDs local to each variety; no gender field. SVQ: about 56 usable speakers (read speech; already used for evaluation) | `census_results.json`, `../llvc_vs_dsp_48spk/PROTOCOL.md` |
| What LLVC training needs | `(PREFIX_original.wav, PREFIX_converted.wav)` pairs made by a teacher that converts every clip to **one** target voice. The original teacher is RVC `f_8312` (a real LibriSpeech speaker) | KoeAI/LLVC README @1627c5d |
| StreamAnon (own model) | training readiness **NO-GO** (`training/readiness.json`): 0 verified GPU hours vs 62–141 A100-h needed; licence blockers; Arabic not ready | `docs/TRAINING_GO_NO_GO.md` |

## Options

| Option | Verdict | Reason |
|---|---|---|
| A. LLVC as-is | **Rejected as a product** | The intelligibility loss is large and robust (two ASRs, two references). The privacy gain over DSP is not established. |
| B. Adapt LLVC to Arabic | **Selected path** | It is the only option with a testable hypothesis at feasible cost: the architecture is already real-time on 1 thread and ONNX-equivalent. The weakness to fix is Arabic intelligibility, which needs Arabic training pairs. |
| C. StreamAnon | Deferred | Its own readiness gate is NO-GO, and it needs a far larger GPU budget than option B. |

## Blockers before any training run (none resolved today)

1. **Teacher ceiling unmeasured (decisive).** The student cannot exceed its teacher. Before training, a teacher must be shown on the frozen evaluation clips to give both:
   - Arabic CER close to the original (proposed gate: relative WER increase ≤ 0.20, the locked `spontaneous_wer_rel`, on both ASRs); and
   - lazy-informed EER ≥ the LLVC values above.

   Candidate teachers:
   - kNN-VC: MIT code, WavLM-Large MIT. Already measured on English to lose intelligibility (`docs/NEURAL_MODEL_EVALUATION.md`); Arabic untested.
   - RVC-type: real-voice target, likeness risk.

   The target voice must be synthetic and licence-clean. MMS-TTS is CC BY-NC and cannot be used.
2. **Consent and permitted use for Omnilingual.** CC BY 4.0 does not license privacy rights. The card states the purpose as ASR and language-ID training, and voice-conversion training is not mentioned.
3. **Privacy advantage not established (G1).** If the gain over DSP is not real, adaptation can at best recover DSP-level privacy at higher cost. The larger evaluation draft (`DECISION_ADDENDUM.md`) resolves this, and is itself blocked on blocker 2.
4. **Compute:** LLVC fine-tune cost on a free T4 is **not measured** (no estimate is given here). The Kaggle free tier currently shows 27.9 h of GPU left this week.

## Next step (needs approval; evaluation only, no training)

A **teacher-ceiling test** on the 225 frozen SVQ clips:
- convert with candidate teacher(s) to a licence-clean synthetic target;
- score with the existing pipeline (2 ASRs, 2 ASVs, paired bootstrap).

Training starts only if a teacher passes blocker 1 **and** blockers 2–3 are resolved.
