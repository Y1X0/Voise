# PROPOSAL (draft, NOT pre-registered, NOT approved): larger Arabic evaluation, LLVC vs DSP-strong

**Status.**
- This draft is for review only.
- Nothing has been downloaded, run or trained for it.
- It becomes a pre-registration only after approval. At that point the manifest and analysis code would be frozen and committed **before** any audio is processed.
- It is a *new* experiment. It is not part of, and cannot amend, the 45-speaker evaluation (`../llvc_vs_dsp_48spk/`, verdict INCONCLUSIVE).

## 1. Question

Is the lazy-informed linkability gain of pretrained LLVC over DSP-strong real? The pre-registered gain threshold is ≥ 0.05 EER on both ASVs. The question also covers how the gain holds against a stronger, semi-informed attacker.

## 2. Data sources (verified 2026-10-10 unless marked)

### google/svq 2.0.0
- **Licence:** CC BY 4.0 (HF card, verified).
- **Speakers:** the four Arabic clean files hold 16, 14, 16 and 15 speakers. 5 of them appear in more than one locale.
  - About 56 single-locale speakers are usable in total. 45 are already used in v1 and 12 were in the exploratory sample.
  - **Effectively exhausted.** Re-use is allowed only as a separately labelled replication set, never pooled as "new" speakers.
- **Reference text:** `transcript_truth` is present.

### facebook/omnilingual-asr-corpus
- **Licence:** CC BY 4.0 (HF card, verified).
- **Speech:** spontaneous answers to survey prompts. Segments are about 90 s.
- **Metadata:** `speaker_id` (local per language, for example `spk03`), `prompt_id`, `segment_id`, and a verbatim `raw_text` with tags such as `<laugh>`.
- **Arabic-variety configs** (verified from the HF dataset viewer):
  - aao, abv, acm, acw, acx, adf, aeb, aec, afb, apc, apd, arq, ars, ary, arz, avl, ayl, ayp;
  - row counts range from 50 (adf) to 3 000 (acw, acx).
- **Speakers per variety: NOT VERIFIED.**
  - The collection targeted about 10 speakers per language; the paper notes that some languages fell short.
  - A partial preview of `apc_Arab` dev showed speakers spk03 and spk05 only.
  - An exact count needs a metadata-only read of the `speaker_id` column. That is a Kaggle CPU job of a few minutes reading no audio, and it needs approval.
  - Upper bound if the target was met: about 18 × 10 ≈ 180 speakers.
- **Unknowns:**
  - whether `speaker_id` is consistent across train/dev/test;
  - whether a speaker can appear under two variety codes.
- **Permitted use:** the card states the purpose as training ASR and language-ID models.
  - CC BY 4.0 permits other uses with attribution.
  - CC BY 4.0 §2(b)(1) does **not** license privacy or publicity rights.
  - The paper says participants were told not to share personal information. It does not describe consent for speaker-recognition research.
  - Proposed handling, as in v1: evaluation only; no audio, converted audio or embeddings leave the compute session; no attempt at re-identification beyond the dataset's own local labels.
- **Redistribution:** allowed with attribution under CC BY 4.0. We would still publish only scores, transcripts of our own ASR, and metadata.

### Not verified in this session

These are listed for completeness only:
- Mozilla Common Voice Arabic: its terms prohibit attempting to identify speakers, which needs legal reading before any speaker-verification use.
- MASC, QASR, ADI-17 and SADA: licences or speaker labels are absent or restricted, per earlier notes.

## 3. Sample size and power

**Variance inputs:**
- Taken from the 45-speaker S3 analysis (`../llvc_vs_dsp_48spk/secondary/uncertainty.json`).
- They are split into a per-speaker part (M1) and a per-clip part (M2 − M1).
- They are scaled by 1/N speakers and 1/(N·c) for c clips per speaker.

**Criterion:**
- 80 % power for "paired 95 % CI lower bound > 0", on **both** ASVs, for the lazy-informed Δ.
- This uses a normal approximation, so it is an estimate, not a guarantee.

| True Δ | M1 (clips fixed) | M2 (two-level), c = 5 | M2, c = 10 | M2, c = 20 |
|---|---|---|---|---|
| 0.09 (observed point) | 87 | 164 | **126** | 106 |
| 0.07 | 143 | 271 | 207 | 175 |
| 0.05 (smallest gain the gate cares about) | 280 | 531 | 406 | 343 |

**Implications:**
- Omnilingual Arabic (at most about 180 speakers, unverified) could resolve a true gain of about 0.09 with c = 10–20 clips per speaker under the conservative two-level CI.
- It **cannot** resolve a true gain of 0.05 under that CI with any free source we have verified.
- If the true gain is near 0.05, a free-data evaluation will remain inconclusive. That outcome itself argues for CHANGE APPROACH rather than more evaluation.

## 4. Design (proposed)

1. **Speakers:** all Omnilingual Arabic-variety speakers with ≥ 10 usable clips, after:
   - deduplication across splits and codes;
   - excluding any speaker who also appears in the attacker-training pool (§6).
   - Target N ≥ 126 with c = 10. If fewer are available, report the shortfall and the corresponding power before running.
2. **Clips:**
   - 4–8 s windows cut from segments with voice-activity detection;
   - at most 2 clips per segment and drawn from at least 3 different prompts per speaker, which limits same-segment and channel leakage;
   - selected by seed; the manifest is frozen and hashed before audio is processed.
3. **Systems:** original, LLVC (PyTorch, chunk ×2, 1 thread), DSP strong. DSP balanced is optional as a reference.
4. **Attackers:**
   - ignorant and lazy-informed, with ECAPA and WavLM-SV as co-primary ASVs (both must pass);
   - top-1 identification over all enrolled speakers;
   - a semi-informed attacker (§6), **only if separately approved**.
5. **Primary uncertainty method** (fixed in advance):
   - M2, the two-level bootstrap with B = 10 000;
   - the decision uses the CI computed at the fixed seed, **and** the Monte-Carlo 95 % interval of the lower bound must lie entirely above 0. A lower bound whose Monte-Carlo interval includes 0 counts as undetermined.
   - M1 and M3 are reported as sensitivity analyses.
6. **Gates:** G1–G3 and G5 as in AUDIT.md §6, unchanged. Intelligibility uses the verbatim reference (`raw_text`, tags removed) with both ASRs, and is reported against the locked `spontaneous_wer_rel` ≤ 0.20.
7. **Dialect:** a descriptive per-variety breakdown. With about 10 speakers per variety there is no per-dialect inference.

## 5. Resource estimate (if approved)

| Step | Compute | Data read | Est. time |
|---|---|---|---|
| Speaker census (metadata only) | Kaggle CPU | `speaker_id`, `prompt_id`, `segment_id`, duration columns only, a few MB | under 10 min |
| Manifest selection (needs segment audio for VAD) | Kaggle CPU | the selected speakers' segments, about 3–6 GB | 30–60 min |
| Evaluation (LLVC 1-thread, DSP, 2 ASR, 2 ASV) | Kaggle T4 | as above | about 2–3 h for about 1 300 clips; LLVC on 1 thread dominates |
| Bootstrap analysis | local CPU | committed scores | about 1 h |

## 6. Informed-attacker plan (needs separate approval: it trains an attacker, not the anonymiser)

**Semi-informed attacker:**
- The attacker knows the anonymisation system.
- It anonymises a **speaker-disjoint** attacker-training pool.
- It fits a scoring backend: LDA + PLDA, or cosine scoring with score normalisation on ECAPA embeddings of anonymised speech. Fine-tuning the ASV is the stronger option.
- It is evaluated on the v2 evaluation speakers.

**Attacker-training pool candidates:**
- the SVQ Arabic speakers (45 + 12; their reuse is acceptable for the *attacker*);
- Omnilingual speakers held out from evaluation.

**Order of work:**
- The backend-only variant runs on CPU in minutes, so it should be done first.
- Fine-tuning is a GPU training job and needs explicit approval.
