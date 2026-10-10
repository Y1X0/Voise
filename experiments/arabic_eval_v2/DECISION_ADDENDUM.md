# Decision addendum to PROPOSAL.md: new, UNAPPROVED pre-registration draft

**Status:**
- This is a draft. It is not frozen and not approved.
- No evaluation has been launched.
- The 45-speaker evaluation verdict stays **INCONCLUSIVE** (`../llvc_vs_dsp_48spk/REPORT.md`, 0ff1e67) and is not amended by anything here.

**Inputs:**
- S4 ONNX validation: `../llvc_vs_dsp_48spk/secondary/onnx_validation_results.json`.
- Metadata census: `census_results.json`.
- S3 uncertainty: `../llvc_vs_dsp_48spk/secondary/uncertainty.json`.

## 1. ONNX (S4)

### OBSERVED

The mounted v1 output held all 450 expected WAVs; ONNX sha256 `e8476add…`. Comparison against the exact evaluated PyTorch audio, through the same PCM_16 encoding:

| Subtest (threshold fixed in ONNX_PLAN.md) | Result | Pass |
|---|---|---|
| Waveform: median SNR ≥ 30 dB, no clip < 20 dB | median 94.4 dB, minimum 85.5 dB, max-abs difference 3.1e-5 (one 16-bit step), no drift | yes |
| Privacy: \|ΔEER\| ≤ 0.02 (2 ASVs × 2 attackers) | largest \|ΔEER\| is 0.0001; same-utterance cosine ≥ 0.99999 | yes |
| Intelligibility: \|Δ median CER\| ≤ 0.02 (2 ASRs) | 0.000 and 0.000. Hypotheses identical: whisper 225/225, MMS 220/225 | yes |

**Verdict: ONNX fp32 is NOT MATERIALLY DIFFERENT from the evaluated PyTorch outputs.**

**Reproducibility:**
- Re-transcribing the retained PyTorch audio gives hypotheses identical to v1: 225/225 for whisper and for MMS.
- Re-embedding it reproduces the committed v1 scores to within 5e-5, which is rounding.

**Timing** (1 thread, Kaggle Xeon 2.0 GHz): p95 12.7 ms per 26 ms chunk, RTF 0.446.

### INTERPRETATION

- The v1 figures (max 0.108, median 0.0015) were very probably a harness artifact. LLVC updates its context buffers in place, and v1 traced the export with the same tensors it then reused as ONNX initial states.
- v1 timing and the G5 result are unaffected.
- ONNX fp32 is suitable for further research and for an Android port.

### LIMITATION

- The cause is inferred from the source code, not re-run.
- int8 was not re-validated. Its v1 difference figures share the same artifact, and it was slower than fp32 in v1.
- No Android device test was run.

## 2. Speaker census (metadata only)

### OBSERVED

- **Source:** facebook/omnilingual-asr-corpus, revision `8648ba89…`; card licence `cc-by-4.0`.
- **Columns read:** language, iso_639_3, glottocode, speaker_id, prompt_id, segment_id, duration. `audio`, `raw_text` and `prompt` were never read.
- **Configs:** 19 ISO-[ara] configs; 18 are `_Arab`, plus `shu_Latn`.
- **Integrity:** 0 missing IDs, 0 duplicate (prompt, segment, speaker) rows. Within every config, no speaker_id appears in more than one split, so the splits are speaker-disjoint.
- **Size:** 184 config-scoped speaker keys and 31 459 rows.
- **Minimum-data rule:** all 184 pass the PRIMARY rule (≥ 5 segments, ≥ 3 prompts, ≥ 120 s). The LOOSE and STRICT rules give the same count, because the smallest speaker has 18.8 min.

Speakers per variety:

| Count | Varieties |
|---|---|
| 20 | acw, aeb, arz |
| 16 | ars |
| 15 | ayl |
| 12 | afb, apd |
| 11 | acx, ayp |
| 10 | apc |
| 8 | abv |
| 7 | aec, arq, ary |
| 3 | acm |
| 2 | avl |
| 1 | aao, adf, shu |

### LIMITATION

- `speaker_id` is a local label (for example `spk03`) scoped to a config. Whether two keys in different varieties are the same person cannot be decided from metadata.
- There is no gender field, so gender cannot be balanced or reported.
- Distinct IDs are assumed, not verified, to be distinct people.

## 3. Feasibility of the speaker-disjoint design

- **Evaluation pool:** up to 184 Omnilingual speakers. Varieties with ≤ 3 speakers are pooled as "other", for descriptive purposes only.
- **Attacker pool** (for the semi-informed attacker, if ever approved): the 57 SVQ speakers. They are disjoint from Omnilingual by construction, because the sources differ. This keeps all 184 Omnilingual speakers available for evaluation.
- **Required N** (80 % power for "paired CI lower bound > 0" on both ASVs, under the primary two-level bootstrap with c = 10 clips per speaker):

| True gain Δ | N needed | Feasible with 184? |
|---|---|---|
| 0.09 (observed) | 126 | **yes** |
| 0.05 (gate minimum) | 406; 343 at c = 20; 280 even under M1 | **no** |

- **Conclusion:** the free data can confirm a gain near the observed +0.09 but cannot establish one near +0.05. This is fixed in advance: a result in between is reported as "not resolvable with available data", **not** as a pass.

## 4. Pre-registration elements (draft)

1. **Primary uncertainty method:** two-level bootstrap (speakers, then clips within a speaker), B = 10 000, fixed seed.
2. **Monte-Carlo stability rule:** a G1 lower bound counts as > 0 only if its order-statistic 95 % Monte-Carlo interval lies entirely above 0. Otherwise G1 is "undetermined".
3. **Sensitivity analyses:** M1 speaker bootstrap and M3 jackknife.
4. **Gates:** G1–G3 and G5 exactly as in AUDIT.md §6, unchanged.
5. **Intelligibility:** verbatim reference (`raw_text`, tags removed), whisper-small + MMS, compared with the locked `spontaneous_wer_rel` ≤ 0.20.
6. **Attackers:** ignorant and lazy-informed with ECAPA and WavLM-SV as co-primary ASVs, plus top-1 identification. The semi-informed attacker is a separate approval (it trains an attacker backend).
7. **Exclusion criteria:**
   - fewer than 10 usable clips after voice-activity detection;
   - clips drawn from fewer than 3 prompts;
   - pre-specified duplicate-speaker screening on original speech: pairs of speaker keys whose original-vs-original ECAPA mean cosine exceeds the 99.9th percentile of the SVQ non-target distribution are flagged and both are excluded. This runs inside the session; nothing is published.
8. **Data protection:**
   - Audio is written only to `/tmp` inside the job, **never to `/kaggle/working`**, so no audio persists in Kaggle outputs. This fixes the cause of the retained audio found in the v1 output.
   - No embeddings, audio or speaker IDs are published. The repository receives aggregates, trial scores and our own ASR transcripts only.
   - Kaggle kernel versions that touched audio are deleted after their logs are retrieved and checksummed.

## 5. Unresolved issues (blocking launch)

1. **Consent and permitted use:**
   - CC BY 4.0 is a copyright licence. It does not license privacy or publicity rights (§2(b)(1)).
   - It does not establish that participants consented to speaker-recognition or voice-privacy research.
   - The card states the purpose as ASR and language-ID training.
   - This must be verified (dataset paper ethics section or terms, or the data owner) before any audio is processed.
2. **Identity semantics:** cross-variety duplicate people are possible (§4.7 is a mitigation, not proof).
3. **No gender metadata.**
4. **Domain shift:** spontaneous, noisy speech vs the SVQ read speech of v1, so results are not directly comparable to v1.
5. **Intelligibility cost:**
   - Pretrained LLVC's loss is already established on two ASRs and two references.
   - A resolved privacy gain would justify only an *adaptation proposal*, never deployment as-is.

## 6. Recommendation: **PROCEED TO NEW PRE-REGISTRATION** (drafting only)

- Freeze this draft only after issue 5.1 (consent and permitted use) is resolved and you approve.
- If the new study rules out a gain of ≥ 0.05, or cannot resolve it, the recommendation becomes **CHANGE APPROACH**.
