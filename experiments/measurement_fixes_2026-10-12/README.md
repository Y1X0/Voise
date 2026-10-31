# Measurement fixes (2026-10-12/13): tie-aware EER, speaker bootstrap, saved scores, encoder licence gate

No model was trained. No recordings were collected. `data/acceptance_criteria.json` is unchanged
(sha256 `58a893c2…1aaf` before and after).

## 1. What changed in the measurement code

| Fix | Where | Test |
|---|---|---|
| Tie-aware EER/AUC: one ROC point per distinct score, EER where FNR = FPR on the piecewise-linear ROC. The old `roc_auc_eer` orders tied scores target-first (all-tied scores give EER 0, AUC 1). It is kept unchanged, because earlier reports used it. | `training/evaluation/privacy_metrics.py` (`roc_auc_eer_ties`) | `training/tests/test_privacy_metrics_ties.py`: edge cases, brute-force and sklearn agreement on tied data, weights = repetition |
| The evaluation and acceptance paths use the tie-aware EER. `evaluation/protocol.py` (A6 scenarios: EER, S1 real-speaker re-ID, CI via `SpeakerBootstrap` with the same draws as before; reports carry `eer_method`) and `evaluation/validator.py` (VALID / HELD_OUT evaluators; `final_eval` goes through it). No committed report was produced by these two modules, so no historical number changes. | `training/evaluation/protocol.py`, `validator.py` | `training/tests/test_eer_method_paths.py`: the legacy function and bootstrap are replaced by raising stubs during real protocol/validator runs; all-tied outputs give EER 0.5 in both; the CI equals the reference bootstrap; the 14 LLVC/DSP EERs are reproduced from the stored scores; a mutant using the legacy function fails 4 tests |
| Speaker bootstrap, vectorised. Same draws as `speaker_bootstrap` (same seed); a trial counts m_a·m_b times, so trials sharing a speaker stay together. 1000 rounds on 460 k trials take about 31 s on CPU. | `privacy_metrics.SpeakerBootstrap`, `percentile_ci` | same file: identical CI to the reference on small data |
| The speaker-encoder evaluation saves every trial score and reproduces its report from the saved files. It splits seen speakers into training-length / held-out utterances and adds 100/200/300-frame crops. | `training/scripts/eval_speaker_encoder.py` | `training/tests/test_speaker_encoder_eval.py` |
| `kaggle.build` used random placeholder encoders whenever an encoder was missing or refused, even with `data_mode="real"`. Placeholders are now allowed only in synthetic smoke mode; real mode raises, and the smoke's dataset check lists the refusal. | `training/trainers/kaggle.py` | `test_encoder_licence_gate.py::test_kaggle_real_mode_never_hides_a_refused_encoder` |
| Encoder licence gate: a role-TRAIN encoder whose `licence_path` the run may not use is refused. The check uses the run's effective path, including `train.py --licence-path`. Missing metadata is refused. | `training/trainers/requirements.py`, `assets.py`, `real_run.py`, `train.py` | `training/tests/test_encoder_licence_gate.py` (a mutant without the check fails 11 assertions) |

## 2. Frozen results recomputed (`historical_recompute.py` → `historical_recompute.json`)

The input is `experiments/llvc_vs_dsp_48spk/scores.json.gz`. These scores are rounded to 4 decimals, so 40–90 % of them are tied.

- `eer_w` from `evaluate.py` reproduces all 14 reported EERs within 0.0001. The residual comes from the rounding of the stored scores.
- The tie-aware EER differs by at most 0.0005 (ECAPA orig↔orig: 0.0484 → 0.0479).
- The paired-difference CIs that decide the verdict keep their signs:
  - WavLM lazy-informed LLVC − DSP-strong: reported [−0.0038, +0.1736]; tie-aware [−0.0049, +0.1738].
  - The lower bound stays below 0, so G1 still fails and the verdict stays **INCONCLUSIVE**.
- No reported result or verdict changes. The reports are not edited.

## 2b. LLVC vs DSP with the corrected measurement (`corrected_protocol_recompute.py` → `.json`)

**HISTORICAL** values are copied unchanged from `experiments/llvc_vs_dsp_48spk/results.json`. That experiment and its REPORT are not modified.

**RECOMPUTED** values use the tie-aware EER on the stored scores, under two speaker-draw sets:
- `exp_boot`: the experiment's own draws, seed 20261011. The only differences from the report are the EER definition and the 4-decimal rounding of the stored scores.
- `spk_boot`: `SpeakerBootstrap`, seed 7.

| Paired LLVC − DSP-strong | Historical | Recomputed (exp_boot) | Recomputed (spk_boot) |
|---|---|---|---|
| ECAPA lazy-informed | +0.0867 [0.0274, 0.1403] | +0.0869 [0.0281, 0.1402] | [0.0233, 0.1410] |
| ECAPA ignorant | +0.1745 [0.1122, 0.2383] | +0.1745 [0.1125, 0.2369] | [0.1188, 0.2361] |
| WavLM lazy-informed | +0.0911 [**−0.0038**, 0.1736] | +0.0911 [**−0.0049**, 0.1738] | [**+0.0017**, 0.1734] |
| WavLM ignorant | +0.0644 [0.0017, 0.1448] | +0.0644 [0.0031, 0.1441] | [−0.0012, 0.1382] |

The 12 per-condition EERs move by at most 0.0002. The 2 original-vs-original EERs are covered in §2: ECAPA moves by 0.0005, WavLM by less than 0.0001.

**Verdict (PROTOCOL.md mapping, applied verbatim):**
- Historical: INCONCLUSIVE.
- Recomputed, pre-registered draws: INCONCLUSIVE (G1 fails; G2, G3 and G5 pass).
- Recomputed, seed-7 draws: G1 would pass mechanically, which would map to CONTINUE.

**Seed sensitivity** (`bootstrap_seed_sensitivity.py` → `.json`):
- Across 30 seeds of 1000 rounds each, the G1-deciding bound (WavLM lazy-informed lower bound) has mean −0.0026 and sd 0.0045. It is above 0 in 27 % of seeds.
- That bound sits within Monte-Carlo noise of 0, so the comparison is underpowered.
- The pre-registered seed governs; the verdict stays **INCONCLUSIVE**.
- A CONTINUE obtained by changing the seed would be a post-hoc choice, not evidence.

## 3. ECAPA held-out evaluation, v2 (`ecapa_vctk_eval_report.json`)

**Run:**
- Kaggle kernel `y1x00r/voise-ecapa-eval-v2`, CPU, 2026-10-12.
- Code: branch commit `43729bd` plus the two working-tree files above, embedded in the kernel and sha256-checked: `privacy_metrics.py 7aea738d…`, `eval_speaker_encoder.py 49f747c4…`.

**Inputs:**
- Model `ecapa_train_inhouse.pt` (sha256 `7d092203…7290`, verified against its .json).
- VCTK from the Kaggle copy `mfekadu/english-multispeaker-corpus-for-voice-cloning`. It uses the 0.80 layout, its provenance is not verified, and it is CC BY 4.0: **research path only, evaluation only**.
- The manifests were rebuilt and are identical to the training run's (`6efc9297…`, `a884a336…`).

**Outputs:**
- `scores.npz` (`fc8863a9…`) and `utterances.jsonl` (`886f2677…`) stay in the kernel output. This environment's network policy blocks `www.kaggleusercontent.com`.
- `report.json` was read from the run log; it is identical to the kernel file (`46bb238b…`).
- `--from-scores` rebuilt the same report inside the run.

EER uses the tie-aware method. The 95 % CI comes from a speaker bootstrap (1000 rounds, seed 7).

| Condition | Speakers | EER | 95 % CI |
|---|---|---|---|
| seen speakers, training-length utterances (≥ 4.015 s) | 61 | 0.0000 | [0.0000, 0.0000] |
| seen speakers, held-out utterances (< 4.005 s) | 61 | 0.0086 | [0.0050, 0.0141] |
| unseen, other half | 43 | 0.0785 | [0.0580, 0.1081] |
| unseen, valid split | 5 | 0.0690 | [0.0100, 0.1693] (too few speakers to use) |
| unseen, all | 48 | 0.0786 | [0.0598, 0.1044] |
| unseen, short utterances (< 4.005 s) | 48 | 0.0800 | [0.0634, 0.1009] |
| length check, 617 unseen utterances ≥ 300 frames: 100-frame crop | 48 | 0.1803 | [0.1427, 0.2196] |
| 200-frame crop | 48 | 0.1032 | [0.0676, 0.1400] |
| 300-frame crop | 48 | 0.0767 | [0.0485, 0.1110] |
| 2-s windows averaged | 48 | 0.0715 | [0.0454, 0.1000] |

**Gaps:**
- unseen − seen, duration-matched (short utterances): +0.0713, CI [+0.0543, +0.0922].
- unseen − seen, training utterances: +0.0786, CI [+0.0598, +0.1044].
- Both CIs exclude 0: on VCTK, the gap between seen and unseen speakers is real.

**Ties:** the cosines are float64, so there are no tied scores here. The legacy and tie-aware EERs agree within 0.0001.

**What this does not show:**
- These numbers measure one embedding on one studio corpus (single session, one microphone). They say nothing about anonymisation.
- The 100-frame result matters for `losses.temporal_consistency`, which calls this encoder on 100-frame halves: at that length its EER is 0.18.
- A paired, utterance-level CI for seen held-out vs seen training utterances (same speakers) is not computed.
