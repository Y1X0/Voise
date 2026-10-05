# Human listening test & real Arabic speech protocol (Phases 2–3)

Status: **tooling ready, NOT YET RUN**. No human listeners and no real Arabic
recordings were available in the environment where this was prepared. The
synthetic "Arabic-like" signals used by the DSP unit tests do **not** count as
Arabic speech evidence.

## 1. Recording (on the phone)

1. Install the APK and plug in wired/USB headphones. Open the **Voice** tab and switch **ON**.
2. Open the **Test** tab → **Blind listening test (A/B/C/D)** → **Record listening-test set**.
3. Speak for 20 s. The first 10 s are kept as the *reference* (real voice); the
   last 10 s become the test sample in four versions:
   A = original, B = Natural, C = Balanced, D = Strong (rendered offline from
   the same take, with the same engine code as the live path).
4. The app writes, only after this button press, to app-specific storage:
   `Android/data/com.voiceanon.app/files/listening-test-<id>/`
   * `reference_original.wav`
   * `sample_1.wav` … `sample_4.wav` (random assignment of A–D)
   * `KEY_experimenter_only.json` (the assignment; keep it away from listeners)
5. Copy to a computer: `adb pull /sdcard/Android/data/com.voiceanon.app/files/ .`

### Arabic material (Phase 3): record one set per speaker and speech type

Use at least **3 speakers** (ideally ≥ 6, mixed male/female, different dialects).
For each speaker record one 20 s set per row. Read the reference half and the
test half with *different* content of the same type.

| Type | Example content (Arabic) |
|---|---|
| Natural conversation | answer freely: «احكِ لي عن يومك أمس» |
| Read text | a paragraph from a news article |
| Numbers | «رقمي صفر سبعة تسعة، أربعة خمسة ستة، واحد اثنان ثلاثة» |
| Names | «محمد، فاطمة، عبد الرحمن، خديجة، يوسف، مريم» |
| Short sentences | «مرحبا، كيفك؟ أنا بخير، شكرًا. وين رايح؟» |
| Fast speech | the read paragraph at maximum comfortable speed |
| Slow speech | the read paragraph slowly and clearly |

Get every speaker's consent for recording and sharing with the listeners. Keep
recordings offline and delete them after the study.

## 2. Listening (blind)

1. Give each listener the session folder **without** `KEY_experimenter_only.json`
   (or with it; the page never reads it, but it is better not to share it).
2. The listener opens `tools/listening-test/index.html` in a browser (works
   offline), enters a participant code, says whether they know the speaker,
   and selects the session folder.
3. The page plays the reference and the four clips in a **second random order
   per listener** ("Clip 1–4"). Neither file names nor the order reveal which
   clip is the original.
4. For each clip they rate on 1–5:
   intelligibility, naturalness, similarity to the reference speaker, artifact
   level (5 = none) and perceived anonymity (5 = could not recognise them).
   They also guess which clip is the unprocessed original.
5. They download `ratings_<code>.csv` and send it to the experimenter.

## 3. Analysis

```bash
python3 tools/listening-test/analyze.py listening-test-*/KEY_experimenter_only.json ratings_*.csv
```

The script prints mean ± 95 % t-interval per condition and criterion, and how
often each condition was taken for the original. With fewer than 10 listeners
it labels the output **EXPLORATORY**. Report it as such: a handful of listeners
gives an indication, not evidence.

Recommended reporting (fill in `docs/REAL_DEVICE_VALIDATION.md`):

* n listeners, n speakers, how many listeners know the speaker
* per condition: intelligibility, naturalness, similarity, artifacts, perceived anonymity
* "guessed as original" rate for A vs B/C/D (chance = 25 %)
* no claims like "unrecognisable"; at most "similarity ratings dropped from X to Y in this sample"

## 4. Speaker-embedding check on the same recordings (Phase 4)

```bash
mkdir -p arabic/orig arabic/proc
# put each speaker's original recordings in arabic/orig/<speaker>_<type>.wav, then:
for f in arabic/orig/*.wav; do for p in natural balanced strong; do
  dsp/build/voiceanon_eval --preset $p --out arabic/proc "$f" >/dev/null; done; done
python3 scripts/speaker_similarity.py --speaker-prefix arabic/orig arabic/proc
```
