# kNN-VC follow-ups, preprocessing, and the Seed-VC Arabic fine-tune (2026-10-24)

Exploratory work on the owner's 52 s recording. Audio, consent records, weights and reports stay under the
git-ignored `runs/` and the session scratch area. Nothing here is an anonymization claim.

Caveat for every Whisper number below: unless stated otherwise, CER/WER compare against Whisper's transcript of
the original, not a human transcript. Identical settings gave 13 % and 47 % CER in two runs, because tiny
neighbour differences change Whisper's hallucinations. Only large differences are meaningful. The reliable
measurements are the ones scored against true transcripts: ClArTTS test and Common Voice test.

## Results

### Preprocessing (`../preprocess_2026-10-24`, `training/scripts/audio_preprocess.py`)

Conversion CER with and without SepFormer-DNS4 + Silero VAD:

| Condition | Without | With |
|---|---|---|
| Clean | 10 % | 9 % |
| Pink noise, 5 dB | 108 % | 58 % |
| White noise, 5 dB | 61 % | 55 % |
| Reverb | 66 % | 66 % |

The enhancer is slow on CPU: about 10 minutes per 52 s.

### Matching-set size (ClArTTS, 10 to 103 min)

No consistent gain beyond about 10 minutes.

### Vocoder fine-tune on ClArTTS (`finetune_vocoder.py`, 3000 steps, CPU)

- ClArTTS test CER, against true transcripts, went from 6.6 % to 5.8 %. This is inside the run-to-run swing (6.3 to 8.2 %).
- The owner preferred the released vocoder by ear. **Not adopted.**

### Dialect (`eval_dialect.py`)

A pooled blend of synthetic Levantine voices (Lahgtna, CC BY 4.0) was compared with the ClArTTS reader.

| Target | CER | WavLM-SV similarity to the owner |
|---|---|---|
| Lahgtna blend | 5.3 % | 0.96 |
| ClArTTS reader | 8.2 % | 0.90 |

- The blend was clearer, but its output moved **toward** the owner.
- The Lahgtna voices are cloned from real speakers whose consent is not documented, so they were used for this test only.

### Prosody (`eval_prosody*.py`)

- Mapping log-F0 to the target lowered SV similarity to the owner only slightly (0.90 to 0.88).
- WORLD post-processing added audible hiss.
- Mapping F0 before conversion removed the hiss but did not lower the similarity.

### Consented target (`convert_to_consented.py`)

The target is a friend who gave written consent; the record is kept under `runs/consent/`. The material is 2.5 minutes of Telegram voice notes.

| System | CER | SV to owner | SV to friend (held-out clips) |
|---|---|---|---|
| kNN-VC, k=4 | 16.6 % | 0.47 | 0.974 |
| Seed-VC (zero-shot, 25 s reference) | **10.7 %** | 0.52 | 0.976 |

- The owner judged both "no longer me, but not yet him".
- Arabic phonemes are weak in both: ح is heard as ه.

### Seed-VC Arabic fine-tune (`../seedvc_arabic_2026-10-24`, Kaggle T4)

- **Data:** Common Voice 18 Arabic (CC0), many speakers, about 40 h, at most 150 clips per speaker.
- **Evaluation:** the Common Voice TEST split, with speakers unseen in training. Each source speaker A is converted using three short clips of speaker B, and Whisper-large-v3 scores the output against the true sentence.
- **Metrics:** CER and WER, Arabic-letter recall (ح ع خ غ ق ص ض ط ظ ث ذ), the rate of ح read as ه, and WavLM-SV similarity to held-out clips of A and of B.
- **Smoke run:** 4 pairs and 40 steps. The pipeline works end to end. The full run's results go here when it completes.
