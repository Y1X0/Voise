# Voise TTS: consent-gated voice cloning in Jordanian/Levantine Arabic

Voise speaks a sentence in a person's voice from a few of their recordings, in Jordanian/Levantine dialect. It also
keeps learning from the voices it is given. It refuses to work without the person's consent, and every file it
makes is marked as synthetic.

It runs on a laptop (Windows + WSL/Ubuntu, CPU only) and on one computer only. It is a graduation project.

## Quick start

```bash
./voise.sh setup                          # once: packages and models (resumes if the download is cut)
# put each person's voice notes in C:\Voise_Voices\<Name>\ with consent.* (and train_ok.* to allow training)
./voise.sh prepare                        # check, convert, and (with train_ok) learn from new voices
./voise.sh serve                          # load the models once and open http://127.0.0.1:8765
./voise.sh speak <person> "جملة باللهجة"   # or: fast <person> "..."
./voise.sh revoke <person>                # consent withdrawn: delete their recordings, outputs and learned voice
```

## How a sentence is made

| Step | What happens | Where |
|---|---|---|
| Consent | Voise refuses unless the person's record is complete (`voice_owner`, `scope`, `revocable`) and not revoked. The record is re-read for every sentence. | `voice_clone_tts.load_consent` |
| Reference | Every 7–12 s stretch of the person's speech is scored on background noise, Whisper's confidence, and how typical it sounds of the person (ECAPA). The best stretch is used. | `select_reference` |
| Text | A paragraph is split at sentence ends and commas into pieces of at most 18 words. Short sentences get enough time so no word is swallowed. Reaction tags `[laughter]`, `[sigh]`, `[surprise-wa]`, `[question-ah]` are supported. | `split_text`, `target_duration` |
| Takes | Up to 3 takes per piece are made with Lahgtna-OmniVoice (+ the learned version). The dialect Whisper transcribes each take, and Voise stops at the first one it hears correctly. Otherwise it keeps the take with the fewest wrong letters, then the closest voice. | `Engine.takes`, `choose_best` |
| Output | `SYNTHETIC_<person>_<time>.wav` is written with an inaudible Perth watermark. A `.json` sidecar records the consent hash, reference, every take's transcript, letter error, voice similarity and UTMOS naturalness. | `Engine.say` |

## Learning loop (`voise_learn.py`)

**Adding voices.** `add` cuts a consenting person's recordings into clean 2–14 s clips. Long continuous talk is cut
at its quietest points. Each clip is checked for background noise and transcribed. A person is added only if their
consent allows training.

**Training.** `train` continues from the current learned version. It trains only the top 6 transformer layers, the
final norm and the audio heads.

**Acceptance.** A new version is kept only if all three hold:
- it fits held-out clips of the known voices better;
- it is not worse on one whole person left out of training (this person rotates by generation);
- Whisper does not hear six test sentences, spoken in two voices, worse.

**Revoking.** `revoke` removes a person from the pool and deletes any learned version that contains their voice.

## Results so far

Each result has a README under `experiments/`.

| Question | Result |
|---|---|
| Better reference choice | Similarity rose from 0.917 to 0.926, and letter error fell from 0.019 to 0.014 (3 speakers). See `ref_select_2026-11-01` |
| Dialect Whisper vs plain Whisper | Letter error 0.083 vs 0.093 on 20 Jordanian clips with known text, and faster |
| Generation 1: 11 min, 6 people | Held-out loss went from 5.90 to 5.19, and letter error from 5.7% to 4.5%. Accepted |
| Generation 2: 38 min, 9 people | Held-out loss went from 5.50 to 5.38. **Unseen person: 5.22 to 5.08.** Letter error went from 2.8% to 0.8%, and naturalness from 2.81 to 2.98. Accepted |
| Restoring WhatsApp notes first (Sidon) | No cloning gain. It made speakers more alike (EER 0.211 to 0.237). See `restoration_2026-10-27` |
| Continued training on 30 h of Syrian speech (Kaggle) | Held-out loss fell, but the accent drifted to Syrian. Not used. See `lahgtna_levant_ft` |
| int8 quantisation | 3.6× faster, but speech broke down (letter error above 1). Not used. See `int8_2026-11-02` |
| Speed | The local engine loads once (27 s instead of about 5 min per run). Prepared voices are cached, and takes stop early |
| Sounding machine-made (Omar) | Forcing 0.6 s/word made speech ~50% slower than he talks. The one clip a listener found real used the model's own timing and spoken-style text. Timing is now left to the model, and a take is accepted early only if Whisper hears every word. See `natural_2026-11-13` |

**Listening test.** `listening_test.py` builds a blind test in which listeners rate each clip 1–5 for "sounds like the
person" and "sounds natural". `score` reports the mean, a 95% confidence interval and head-to-head wins.

## Safety and licences

**Consent and privacy.**
- Every person's consent is recorded under `runs/consent/`.
- Audio, consent records, models and learned versions live under the git-ignored `runs/` and `models/` folders and
  are never committed.
- The local engine accepts requests only from its own page on 127.0.0.1. It checks the Host header, the JSON content
  type and the Origin.
- Outputs are watermarked (Perth) and named `SYNTHETIC_*`.

**Licences.**
- **Lahgtna-OmniVoice weights state no licence.** Use is private or academic until the author permits more.
- OmniVoice code: Apache-2.0.
- The Whisper dialect fine-tune: Apache-2.0.
- ECAPA: Apache-2.0.
- UTMOS (SpeechMOS) and Perth: MIT.

**Out of scope.** Publishing anyone's voice, speaking as someone to other people, and using recordings of people who
did not agree, such as public videos or broadcasts with guests.

## Files

| File | Role |
|---|---|
| `voise.sh` | Laptop entry point |
| `training/scripts/voice_clone_tts.py` | Engine: consent, reference choice, takes, output |
| `training/scripts/voise_server.py`, `voise_ui.html` | Local engine and demo page |
| `training/scripts/voise_learn.py` | Learning loop |
| `training/scripts/listening_test.py` | Blind listening test |
| `training/tests/test_voice_clone_tts.py`, `test_voise_learn.py`, `test_voise_server.py`, `test_listening_test.py` | Tests (no model needed) |

## Since 2026-11-13

- **Pronunciation dictionary and vowel check** (`pronlex.py`). Whisper cannot hear a wrong vowel. A phoneme
  recogniser learns each word's first vowel from the consented recordings. The dictionary is cumulative: 810
  words after two Mafia episodes. A take with a wrong vowel is retried.
  - Example (Omar): take 1 said «يُطلع» for «يِطلع» and was rejected.
  - `pronunciation.json` swaps words the model keeps getting wrong: ومنحكي → وبنحكي was right 2/2, against 0/2
    as written.
- **Learning keeps everything.** Each generation continues from the last; the last three are archived in
  `runs/learn/history/`. Generation 2: letter error 0.86%, unseen person's loss 5.31.
- **Say it yourself** (`voise_convert.py`, kNN-VC). Your recording, in a consenting person's voice, with your own
  laughs and emotion. Same gates as the TTS tool.
  - Test, Omar → عبود: similarity to عبود 0.34 → 0.68, to Omar 0.91 → 0.52; letter error vs the source 5.7%.
- **Encrypted backup** of the model and dictionary in `backup/`.
