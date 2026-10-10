# Pre-registered protocol: pretrained LLVC vs shipped DSP on Arabic

This protocol was written, and `manifest.json` was frozen, **before** `evaluate.py` produced any result.

- Evaluation only. No training, no fine-tuning, no adaptation.
- `data/acceptance_criteria.json` is unchanged (sha256 prefix 58a893c2).
- `app/`, `dsp/` and `termux/` are unchanged.

## Sample (`manifest.json`, sha256 `9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685`)

Selected by `select_manifest.py` (Kaggle kernel, seed 20261011) from google/svq 2.0.0, `clean` condition.

**Design:**
- 4 Arabic locale groups, 12 speakers each, 5 clips per speaker, each clip ≥ 1.5 s;
- only speakers who appear in exactly one Arabic locale (across all four conditions);
- the 12 speakers of the earlier exploratory and audit sample are excluded, so this is a fresh hold-out;
- no two clips share a prompt text.

**Shortfall (recorded, not filled):**

| Locale | Speakers in clean file | In > 1 locale | Earlier sample | Eligible = selected | F / M / no answer |
|---|---|---|---|---|---|
| ar_eg | 16 | 4 | 3 | 11 | 3 / 8 / 0 |
| ar_x_gulf | 14 | 0 | 3 | 11 | 9 / 1 / 1 |
| ar_x_levant | 16 | 0 | 3 | 12 | 9 / 2 / 1 |
| ar_x_maghrebi | 15 | 1 | 3 | 11 | 4 / 7 / 0 |
| **total** | | | | **45** (target 48) | 25 / 18 / 2 |

- 225 clips, 963 s in total.
- Gender balance within a locale is not achievable with this source.
- Locale and gender are partly confounded (Gulf and Levantine are mostly female), so per-locale numbers are descriptive only.

## Systems (same 225 clips)

- `orig`
- `llvc`: KoeAI/LLVC @1627c5d, `G_500000.pth`, streaming chunk ×2 on 1 CPU thread. This is the deployable setting; the earlier runs used chunk ×1.
- `dsp_strong`: the decision baseline. `dsp/tools/eval.cpp --preset strong --render-only`, built from this commit.
- `dsp_balanced`: reference only.

## Measurements

**Speaker verification (ASV):**
- Models: SpeechBrain ECAPA-VoxCeleb and WavLM-base-plus-sv.
- Scores are cosine similarities; a trial is accepted if its score ≥ the threshold.
- Trial lists:
  - **original vs original** (sanity check): unordered pairs of different clips;
  - **ignorant attacker**: enrolment is the *original* clip i, trial is the *processed* clip j, all ordered pairs i ≠ j;
  - **lazy-informed attacker**: processed vs processed, unordered pairs of different clips.
- The two attacker scenarios are reported separately and never pooled.
- EER uses pooled trials over all 45 speakers.
- 95 % CI: speaker bootstrap, 1000 resamples.
- The paired difference (system − dsp_strong) uses the same resamples.
- Also reported:
  - effective EER = min(EER, 1 − EER);
  - FAR/FRR at the threshold calibrated on original speech;
  - EER with same-locale non-target trials only (secondary);
  - top-1 identification over all 45 enrolled speakers. Each enrolment is the mean of the speaker's clips, leaving out the trial's own utterance. The enrolment is original speech for the ignorant attacker and processed speech for the lazy-informed attacker.

**Speech recognition (ASR):**
- Models: whisper-small (Arabic) and MMS-1B-all with the `ara` adapter. MMS-1B-all is CC BY-NC, so it is used for evaluation only.
- Text normalisation is the same as in the audit.
- Metrics:
  - CER (median);
  - corpus WER;
  - relative WER increase, with a speaker-bootstrap CI;
  - an ASR-reliable subset (original CER ≤ 0.15 for that ASR);
  - CER against the original's own hypothesis;
  - a paired LLVC − DSP-strong comparison.

**Speed:**
- PyTorch LLVC on 1 thread: chunk ×2 on all clips, and chunk ×1 on every 4th clip.
- ONNX export of one streaming step, run on ONNX Runtime with 1 thread, in fp32 and dynamic int8. The output is checked against PyTorch.
- DSP real-time factor (RTF) on the same CPU, pinned to one core.

**Android:** no physical device is available, so no on-device result can be claimed. The ONNX export only shows that a deployable graph exists.

## Decision rule

The numeric gates are copied unchanged from `experiments/llvc_arabic_svq/AUDIT.md` §6, as committed at 49562de before this run.

| Gate | Pass condition |
|---|---|
| G1 linkability gain | lazy-informed EER(LLVC) − EER(DSP-strong) ≥ 0.05 **and** paired 95 % CI lower bound > 0, on **both** ASVs |
| G2 linkability level | lazy-informed EER(LLVC) ≥ 0.20 and CI lower bound ≥ 0.15, on both ASVs |
| G3 ignorant attacker | EER(LLVC) ≥ 0.35 on both ASVs. Raw and effective EER must both pass, because an EER above 0.5 only means the scores are inverted. |
| G4 intelligibility | No gate for the pretrained model; recorded as the baseline. The locked final target is a relative WER increase ≤ 0.20 (`spontaneous_wer_rel`). |
| G5 real time | ONNX, chunk ×2, 1 thread: p95 < 26 ms and no lag build-up (final lag of every clip < one chunk) |

**Verdict mapping** (fixed now):
- **CONTINUE** to an adaptation *proposal*: G1, G2, G3 and G5 all pass. Training would still need separate approval.
- **REJECT**: G1 fails *and* the gain is ruled out. "Ruled out" means, on at least one ASV, the paired-difference CI upper bound is < 0.05, or the point estimate is ≤ 0.
- **INCONCLUSIVE**: G1 fails, but on both ASVs the paired CI still includes a gain ≥ 0.05 (an underpowered comparison). The same applies if G1 passes while G2, G3 or G5 cannot be evaluated.
- INCONCLUSIVE is not success and authorises no training.
- If G1 passes but G2, G3 or G5 fails, the verdict is **REJECT for adaptation as-is**, with the failing gate named.

## Privacy and redistribution

- Audio, ASV embeddings and model outputs stay inside the Kaggle session and are not published.
- The repo receives:
  - this protocol;
  - the manifest (pseudonymous SVQ IDs, gender, prompt text);
  - the summary;
  - per-clip ASR transcripts and error rates;
  - trial scores. These are cosine numbers, from which no voice can be reconstructed.
