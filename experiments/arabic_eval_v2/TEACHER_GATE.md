# Teacher Feasibility Gate: pre-registration (written before any teacher output exists)

- **Purpose:** decide whether any eligible teacher can produce Arabic `(original, converted)` pairs worth training LLVC on.
- **Scope:** evaluation only. No training, no change to the frozen evaluation data, and no change to the locked criteria.

## Candidate eligibility (checked before the run)

| Candidate | Code / weights licence | Target voice | Eligible |
|---|---|---|---|
| **kNN-VC** (`bshall/knn-vc` @ c616845c) | MIT (repo LICENSE). Weights `prematch_g_02500000.pt` and `WavLM-Large.pt` come from the repo's own release v0.1 (`docs/NEURAL_MODEL_ACQUISITION.md`) | Synthetic pseudo-speaker, see below | **yes** |
| VoicePrivacy B3 | GPL-3.0 code; component model licences NOT VERIFIED | – | no (licence unverified) |
| RVC (LLVC's original teacher `f_8312`) | – | A real person's voice (LibriSpeech speaker 8312), no consent for cloning | no |
| MMS-TTS | CC BY-NC 4.0 | – | no (non-commercial; also TTS, not voice conversion) |

**Synthetic target voice:**
- **Source pool:** the 12 SVQ speakers of the earlier exploratory sample, 60 clips.
  - SVQ is CC BY 4.0.
  - These speakers are speaker-disjoint from the 45 test speakers; overlap 0 was verified in the review.
  - Their audio is mounted read-only from the retained Kaggle output of `y1x00r/llvc-svq-audit`. **No new audio is downloaded.**
- **Transform:** each pool clip goes through the repo's WORLD pseudo-speaker `PseudoSpeakerWorld(cmvn=True)` (`scripts/neural_anonymization_eval.py`).
  - A single fixed seed (4242) gives **one** synthetic voice, as LLVC training requires.
  - The resulting renders form the kNN-VC matching set.
- **Limitation:** the renders derive from 12 real (CC BY) speakers. How much of their identity survives is measured under "pool attribution" below. A synthetic target does **not** by itself guarantee unlinkability.

## Test set and measurements

**Test set:**
- The 225 frozen clips, 45 speakers (`../llvc_vs_dsp_48spk/manifest.json`, sha256 `9cca9b92…`).
- They are mounted read-only from `y1x00r/llvc-dsp-48spk-eval` v1.
- Teacher output is kept in memory or `/tmp` only and is never written to `/kaggle/working`.

**Pipeline:** the same as `evaluate.py`.
- **ASR:** whisper-small (ar) and MMS-1B-all (ara), with the same normalisation.
- **ASV:** ECAPA-VoxCeleb and WavLM-base-plus-sv.
- **Trials:** the same lists, with DSP-strong and LLVC scores taken from the committed `scores.json.gz`.

**Reported separately (never pooled):**
1. **Intelligibility:**
   - relative corpus-WER increase vs the original, for both ASRs;
   - primary reference: prompt text, as in v1; secondary reference: `transcript_truth`;
   - speaker-bootstrap 95 % CI.
2. **Similarity to the original speaker (ignorant attacker):** EER, mean same-utterance cosine, top-1 identification.
3. **Distinctness of converted speakers (lazy-informed attacker):** EER, top-1 identification among converted outputs.
4. **Pool attribution:** mean cosine of teacher outputs to the 12 pool speakers' originals, vs to unrelated test-speaker originals.
5. **Speed:** GPU time per clip, and CPU RTF on 1 thread on every 9th clip (25 clips). The teacher is offline, so this is reported but is not a gate.

**Uncertainty:**
- Speaker bootstrap, B = 10 000, seed 5001.
- Paired with the same resamples for teacher, DSP-strong and LLVC.
- Two-level (speaker + clip) bootstrap as a sensitivity analysis, seed 5002.

## Gates (fixed now; thresholds copied unchanged from existing documents)

| Gate | Pass condition | Source of threshold |
|---|---|---|
| T1 intelligibility | relative corpus-WER increase ≤ 0.20 on **both** ASRs (point estimate, prompt reference) | locked `spontaneous_wer_rel` (`data/acceptance_criteria.json`) |
| T2 privacy gain | lazy-informed EER(teacher) − EER(DSP-strong) ≥ 0.05 **and** paired speaker-bootstrap CI lower bound > 0, on both ASVs | G1, AUDIT.md §6 |
| T3 ignorant attacker | EER(teacher) ≥ 0.35 on both ASVs | G3 = locked `strong_pretrained_attacker_eer` |
| T4 licence | teacher code and weights permissive; target synthetic; pool data CC BY 4.0 | eligibility table above |

**Verdict mapping:**

| Verdict | Condition |
|---|---|
| **TEACHER PASS** | T1–T4 all pass |
| **TEACHER FAIL** | T1 fails (the student cannot exceed its teacher's intelligibility), **or** T2's gain is ruled out (paired CI upper bound < 0.05 or point ≤ 0 on either ASV), **or** T3 fails |
| **BLOCKED** | the test cannot run (missing weight, dependency or permission), **or** T1 and T3 pass but T2 is undetermined |

A PASS permits only *planning* training. The other training preconditions remain:
- Omnilingual consent / permitted use;
- a measured training cost.
