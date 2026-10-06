# Neural model offline evaluation (Phase 3)

**Classification: B — PARTIALLY_VALIDATED.**

Speaker *replacement* towards a synthetic pseudo-speaker (kNN-VC, VoicePrivacy B3)
greatly reduces linkability against an informed attacker. The result holds across all
four speaker evaluators and all three cross-session pairs, but it costs a large part of
intelligibility. Neither model can run in real time on a phone, and no human listening
test was run.

The current app's DSP is confirmed **not** to anonymize against an informed attacker.

Arabic: **ARABIC_NOT_VERIFIED**. No human listening test was run. Android: nothing was
ported, by rule; see `NEURAL_ANDROID_FEASIBILITY.md`.

The DSP engine, Android app, Termux CLI and call integration were not modified.

Raw data:
* `docs/results/phase3_model_evaluation.json`
* `docs/results/phase3_model_evaluation_tables.md`, with every CI and every attack

Artifacts and their provenance: `NEURAL_MODEL_ACQUISITION.md`.

## 1. What was run

| Item | Role | Weights | Status |
|---|---|---|---|
| GE2E (Resemblyzer) | evaluator 1, as before | bundled with the pip package | run |
| MFCC statistics | evaluator 2 (classical, independent), as before | none | run |
| **VPC 2024 ASV `asv_orig`** (ECAPA-TDNN, LibriSpeech-360) | evaluator 3 (modern; the official VoicePrivacy evaluator) | official GitHub release | run |
| **SA-toolkit ResNet `resnet_v1`** (VoxCeleb1, augmented) | evaluator 4 (modern; different architecture and data) | official GitHub release | run |
| SpeechBrain ECAPA (VoxCeleb), WavLM-Base-Plus-SV | the evaluators named in the request | huggingface.co blocked (403) | **NOT_VERIFIED**, replaced by evaluators 3 and 4 |
| **kNN-VC → synthetic pseudo-speaker** | candidate | official GitHub release | run |
| **VoicePrivacy 2024 B3** (STTTS + GAN artificial speaker) | candidate | official GitHub release | run |
| Whisper small.en (ONNX) | intelligibility proxy | official sherpa-onnx GitHub release | run |

The protocol is unchanged: same corpus, split (`docs/results/neural_splits.json`),
seeds, attacks, 1000 speaker-level bootstrap resamples and acceptance criteria.

* **Corpus:** 15 real speakers. Only the 9 unseen TEST speakers (27 utterances) are scored.
* **Sessions:** A/B/C = seeds 101/202/303.

No parameter was chosen on test data:
* kNN-VC uses the authors' default `topk=4`.
* Its matching set (`psn_world_cmvn` renders of the TRAIN speakers) was fixed before any run.
* B3 uses the unmodified VPC 2024 configuration.

### Target voices (no impersonation)

**kNN-VC.** The matching set for session S is the WORLD pseudo-speaker rendering of the
4 TRAIN speakers (12 utterances, 28–30 s). That rendering uses one synthetic voice drawn
per session seed, and no test speaker is in it.

* The authors' VAD trim removed whole files for 6–7 of the 12 references; those were
  used untrimmed and logged.
* Target-attribution check (section 4): the output does not match any of those real
  TRAIN speakers at same-speaker level.

**B3.** Every utterance gets a new artificial speaker embedding sampled from the VPC
WGAN, from a pool of 5000 vectors per session.

## 2. Privacy against the informed attacker

The informed attacker (*lazy-informed*) knows the system. They anonymize their own
enrollment recordings with it and compare processed↔processed.

* *Cross-session* means enrollment and test come from different sessions, i.e. a
  different pseudo-speaker draw.
* Each cell is the A→B EER with its 95 % CI, followed by A→C / B→C.
* Chance is EER 50 %, top-1 11 % (9 speakers).

| System | GE2E | MFCC-stats | **VPC ECAPA** | **ResNet-Vox1** |
|---|---|---|---|---|
| RAW (original↔original) | 5.1 % [0–9], top-1 93 % | 20.0 % [3–28], 74 % | **0.6 % [0–4], 100 %** | **0.3 % [0–5], 96 %** |
| DSP Natural | 6.3 % [0–11] · 6.4 / 6.1 | 28.2 % · 28.1 / 28.2 | 3.6 % [0–6] · 2.3 / 2.5 | 2.6 % [0–6] · 2.6 / 2.6 |
| DSP Strong | 3.8 % [0–12] · 3.8 / 5.1 | 27.0 % [10–39] · 25.6 / 25.9 | **1.3 % [0–3]** · 1.4 / 2.5 | **1.2 % [0–4]** · 1.3 / 1.4 |
| WORLD pseudo-speaker | 14.1 % [4–22] · 14.4 / 11.5 | 20.5 % · 20.8 / 20.7 | 2.9 % [0–8] · 3.8 / 2.6 | 3.7 % [0–9] · 2.6 / 2.6 |
| WORLD variance-normalized | 14.2 % [4–20] · 15.4 / 12.8 | 28.4 % · 29.6 / 28.4 | 7.5 % [0–13] · 6.4 / 9.1 | 5.0 % [1–14] · 5.1 / 5.1 |
| **kNN-VC → pseudo-speaker** | **33.6 % [24–43]** · 35.9 / 30.8 | **42.5 % [31–49]** · 44.9 / 43.6 | **30.9 % [18–41]** · 30.9 / 28.3 | **34.6 % [28–40]** · 33.3 / 34.7 |
| **VPC B3** | **47.4 % [40–55]** · 42.3 / 47.4 | **50.0 % [46–63]** · 51.3 / 47.4 | **45.0 % [37–50]** · 41.2 / 42.1 | **45.2 % [40–53]** · 44.7 / 49.8 |

Cross-session A→B top-1 identification (chance 11 %):

| System | GE2E | MFCC | VPC ECAPA | ResNet |
|---|---|---|---|---|
| DSP Strong | 93 % | 67 % | 100 % | 96 % |
| WORLD pseudo-speaker | 81 % | 63 % | 100 % | 93 % |
| kNN-VC | 30 % | 15 % | 26 % | 41 % |
| VPC B3 | 15 % | 15 % | 26 % | 22 % |

The adaptive attacker (WCCN back-end trained on TRAIN+VAL processed audio) changes little:

* kNN-VC: 28.5 % (VPC ECAPA) and 35.9 % (ResNet);
* B3: 44.7 % and 46.1 %.

The same-session attack (enrollment and test from the same session) gives similar numbers.

**OBSERVED FACT.** Under both modern evaluators, every DSP preset and both WORLD
variants stay almost fully linkable: EER 1–9 %, top-1 93–100 %.

* The GE2E-only improvement of WORLD seen in the previous phase does not survive a modern
  evaluator.
* Pitch/formant transformation (the app) is **not anonymization** against this attacker.

**OBSERVED FACT.** kNN-VC and B3 raise the informed-attacker EER under all four
evaluators, in all three session pairs.

* Their CIs do not overlap DSP Strong's under GE2E, VPC ECAPA and ResNet.
* Under the weak MFCC evaluator, kNN-VC's CI touches DSP Strong's ([31–49] vs [10–39]);
  B3's does not.
* There were no VAD failures for either model.

**LIMITATION (important).** Our strongest attacker is lazy-informed + WCCN. The official
VPC 2024 results for B3 (`models/vpc2024/results/result_for_rank_sttts`, LibriSpeech, an
ASV *retrained on B3-anonymized speech*, i.e. semi-informed) report an EER of only
**22.0–28.4 %**.

* Our 42–50 % for B3 therefore overstates its privacy against a capable attacker.
* kNN-VC was not tested against a retrained ASV either.
* With 9 test speakers the CIs are wide.

## 3. Intelligibility and quality (TEST, session A)

Relative WER means the ASR transcript of the original is the reference, since the
corpus has no transcripts. Whisper is the meaningful column; pocketsphinx is the old,
weak proxy, kept for continuity.

| System | **WER (Whisper small.en)** | WER (pocketsphinx) | ESTOI | DNSMOS OVRL / SIG | speech dropouts | duration ratio | clipped |
|---|---|---|---|---|---|---|---|
| DSP Natural | 8.9 % | 55 % | 0.76 | 2.58 / 2.92 | 0.1 % | 1.000 | 0 |
| DSP Balanced | 8.1 % | 60 % | 0.64 | 2.59 / 2.93 | 0.1 % | 1.000 | 0 |
| DSP Strong | 10.0 % | 62 % | 0.54 | 2.56 / 2.90 | 0.1 % | 1.000 | 0 |
| WORLD pseudo-speaker | 9.3 % | 40 % | 0.69 | 2.70 / 3.12 | 0.0 % | 1.000 | 0 |
| WORLD variance-normalized | 20.5 % | 51 % | 0.66 | 2.59 / 2.99 | 7.2 % (pyannote files destroyed) | 1.000 | 0 |
| **kNN-VC** | **34.4 %** | 63 % | 0.45 | 2.61 / 3.03 | 0.0 % | 0.994 | 0 |
| **VPC B3** | **44.8 %** | 68 % | 0.15¹ | 2.97 / 3.37 | 5.8 %¹ | 0.956 | 0 |

¹ B3 re-synthesizes speech with new durations, so ESTOI and the dropout measure (both
need time alignment) are not meaningful for it.

Whisper WER by recording type (kNN-VC / B3 / DSP Strong):

| Recording type | kNN-VC | B3 | DSP Strong |
|---|---|---|---|
| AMI meetings (198 words) | 34 % | 44 % | 7 % |
| CMU Arctic read speech (21 words) | 48 % | 33 % | 19 % |
| pyannote (32 words) | 31 % | 62 % | 25 % |

The loss is not limited to the noisy meeting audio.

**OBSERVED FACT.** Both anonymizing models lose a large share of words. Roughly 1 word
in 3 (kNN-VC) or 1 in 2 (B3) differs from what the same recognizer heard in the
original, against about 1 in 10 for DSP.

* DNSMOS (naturalness proxy) is equal or better.
* B3's official LibriSpeech WER is 4.3 % (vs 1.8 % original). That was clean read speech
  with an in-domain ASR, so the corpus and recognizer matter a lot.

**INTERPRETATION.** On this corpus, "the speech stays clear" is not met by either model.

kNN-VC's matching set is only ~29 s; the authors recommend minutes. That probably costs
intelligibility. It was not changed, because tuning it on these results would be test-set
tuning (see section 7).

## 4. Target attribution: does the output impersonate a real person?

For each processed TEST utterance, the evaluator measures the highest cosine similarity
to any TRAIN speaker's original voice (`scripts/target_attribution.py`). These are the
real people whose WORLD-transformed recordings form kNN-VC's matching set.

| Evaluator | kNN-VC | B3 | DSP Strong | Unrelated real voice (reference) | Same speaker (reference) |
|---|---|---|---|---|---|
| GE2E | 0.667 | 0.640 | 0.679 | 0.689 | 0.857 |
| VPC ECAPA | 0.358 | 0.154 | 0.259 | 0.324 | 0.706 |
| ResNet-Vox1 | 0.231 | 0.105 | 0.151 | 0.169 | 0.604 |

**OBSERVED FACT.** kNN-VC output sits slightly above an unrelated real voice under the
two modern evaluators, but far below same-speaker level. It is not a clone of any TRAIN
speaker. B3 is the least similar to any real person.

The script also prints the share of utterances above the same-speaker 5th percentile.
That threshold falls below the unrelated-voice level for the modern evaluators, so the
column flags even DSP output and is not informative. The means above are the evidence.

## 5. Performance

| System | Size | Measured compute (4-core Xeon 2.1 GHz) | Peak RAM | Streaming? |
|---|---|---|---|---|
| App DSP | none | RTF 0.010 (incl. process start) | small | yes, 32 ms algorithmic latency |
| kNN-VC | 332 M params (1.26 GB WavLM-Large + 66 MB HiFi-GAN, fp32) | RTF 0.50 on 1 thread, 0.16 on 4 threads (load excluded) | 3.0 GB | no: WavLM attends over the whole utterance; non-causal kNN over frames |
| VPC B3 | ≈1.2 GB (ASR + FastSpeech2 + HiFi-GAN + aligner + GAN) | RTF 0.92 on 4 threads, incl. model load (3.8–3.9 while the CPU was shared) | ~1 GB+ | no: ASR → TTS on the whole utterance |

## 6. Answers

* **Naive (ignorant) attacker.** Enrollment is original speech, test is processed speech.
  * DSP Strong is partial: EER 18–22 % on the modern evaluators, vs 0.3–0.6 % for raw speech.
  * kNN-VC: 29–47 %.
  * B3: 44–50 %.
* **Informed attacker.**
  * DSP: **no**. EER 1–4 %, top-1 96–100 % on the modern evaluators.
  * kNN-VC: **partial**. EER 28–36 % on the modern evaluators, top-1 26–41 %, against a
    chance level of 11 %.
  * B3: **strong against our attackers, but the official semi-informed attacker reaches
    22–28 % EER**.
* **Intelligibility.**
  * DSP and WORLD: yes (Whisper relative WER 8–10 %).
  * kNN-VC and B3: degraded (34 % and 45 %).
  * Human listening: NOT_VERIFIED.
* **Real-time on Android:** no for both models (`NEURAL_ANDROID_FEASIBILITY.md`).
* **Arabic:** ARABIC_NOT_VERIFIED. There are no Arabic recordings, and Whisper small.en
  is English-only.

## 7. Acceptance criteria (unchanged) for kNN-VC

| Criterion | kNN-VC | B3 |
|---|---|---|
| Lower linkability under every evaluator | yes (4/4) | yes (4/4) |
| CIs exclude the baseline | yes for 3/4; MFCC overlaps | yes (4/4) |
| Top-1 drops | yes (93–100 % → 15–41 %) | yes (→ 15–26 %) |
| Low VAD/embedding failure rate | yes (0) | yes (0) |
| Quality does not collapse | **no** (Whisper WER 34 % vs 10 %; ESTOI 0.45) | **no** (WER 45 %) |
| Repeats across sessions A/B/C and WCCN | yes | yes |
| No test-set tuning | yes | yes |
| Stronger (semi-informed) attacker | not run | **fails** in the official results (22–28 %) |
| Real-time on the phone | no | no |

The privacy criteria are met; the utility and deployability criteria are not. Hence
**B — PARTIALLY_VALIDATED**, not A. It is not C: the privacy gain is large, consistent
across evaluators, and outside the CIs. It is not D: models were obtained and run.

## 8. Limitations

* 15 speakers / 9 TEST speakers / 27 test utterances; the CIs are wide.
* No semi-informed attacker: retraining an ASV needs far more speakers than the 6
  TRAIN+VAL speakers.
* No human listening test; no Arabic.
* SpeechBrain VoxCeleb ECAPA and WavLM-SV were unobtainable (Hugging Face blocked by the
  egress policy). Two other modern ASVs replace them, so the requested set is not
  complete.
* The relative WER uses Whisper on the original as the reference, not human transcripts.
* RTF values from the main run include per-file process start and model loading; the
  table in section 5 uses the separate benchmark.

## 9. Reproduce

```bash
scripts/fetch_models.sh                       # official sources, SHA-256 pinned
TORCH_HOME=models/torch_hub HF_HUB_OFFLINE=1 python3 scripts/neural_anonymization_eval.py \
  --out eval-neural-p3 --vpc-asv-dir models/exp/asv_orig \
  --satools-asv-jit models/satools_resnet_v1/final.jit \
  --systems "dsp_natural,dsp_balanced,dsp_strong,psn_world,psn_world_cmvn,knnvc_pseudo=cmd:python3 scripts/model_adapters/knnvc_pseudo.py --repo models/knn-vc --ref-root eval-neural-p3/psn_world_cmvn --in {in} --out {out} --seed {seed}"
models/venv_b3/bin/python scripts/model_adapters/vpc_b3_render.py --vpc models/vpc2024 --out eval-b3
python3 scripts/neural_anonymization_eval.py --out eval-neural-p3-b3 --vpc-asv-dir models/exp/asv_orig \
  --satools-asv-jit models/satools_resnet_v1/final.jit \
  --systems "vpc_b3=cmd:python3 scripts/model_adapters/prerendered.py --root eval-b3 --in {in} --out {out} --seed {seed}"
python3 scripts/whisper_wer.py --model-dir models/sherpa-onnx-whisper-small.en --systems eval-neural-p3/<system> ...
python3 scripts/target_attribution.py --out eval-neural-p3 --systems knnvc_pseudo,vpc_b3,psn_world_cmvn,dsp_strong \
  --vpc-asv-dir models/exp/asv_orig --satools-asv-jit models/satools_resnet_v1/final.jit
TORCH_HOME=models/torch_hub python3 scripts/bench_knnvc.py --repo models/knn-vc \
  --wav eval-corpus/arctic-axb-female__0.wav --matching eval-neural-p3/psn_world_cmvn/A/_knnvc_matching_set.pt
```

## 10. The one thing that would move this to a higher level

A **small, causal, streaming speaker-replacement model, trained for this purpose**:
* a distilled content encoder;
* a synthetic pseudo-speaker embedding;
* a light causal vocoder, int8, a few tens of MB.

It must then pass three checks: a semi-informed attacker, Whisper WER near the DSP's,
and a human listening test.

That needs a GPU training run on a multi-speaker corpus (e.g. LibriSpeech/LibriTTS),
which cannot be done in this 4-core CPU container. The design and budget are in
`NEURAL_ANDROID_FEASIBILITY.md`.
