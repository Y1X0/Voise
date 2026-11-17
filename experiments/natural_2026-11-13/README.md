# Why the clone sounds machine-made: pace, room sound, guidance (2026-11-13)

**Why.** Yazan: every clip is obviously AI to anyone who hears it, even with correct words. This tests the
settings that shape delivery rather than words, on Omar (consented, ~23 min in the learning pool) with his
round-1 personal version (`../personal_2026-11-07`).

**Conditions.** Same voice version, reference, seed 0 and temperature 0.5; three sentences, one take each.
- `A` current tool: sentences of up to 10 words forced to 0.6 s/word + 1 s; guidance 2.0; denoise on.
- `B` his own pace: duration = characters × his median seconds per character (0.0986, from his 161 real clips) × 1.05.
- `C` B + denoise off: the model keeps the reference's room sound instead of studio-clean audio.
- `D` C + guidance 1.5.

| | pace (s/char) | pitch movement (F0 std, semitones) | letter error | similarity | UTMOS |
|---|---|---|---|---|---|
| Omar, real clips | **0.099** | **4.7** | | | 2.16 |
| A current | 0.145 | 4.80 | 0.016 | 0.707 | 2.93 |
| B own pace | 0.108 | 3.63 | 0.016 | 0.673 | 2.95 |
| C own pace + room | 0.108 | 4.13 | 0.025 | **0.711** | 2.71 |
| D own pace + room + guidance 1.5 | 0.108 | 3.53 | 0.022 | 0.683 | 2.57 |

**Reading.**
- The current tool speaks about 50% slower than Omar does. «ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟»
  lasted 7.2 s; at his pace it is 3.8 s. Slow, even delivery is a likely reason the clips sound read out.
- Matching his pace keeps the words intact (same letter error) but flattens pitch movement and lowers similarity.
- Turning denoise off on top of it restores pitch movement and gives the best similarity of all four. UTMOS drops
  towards his real recordings' level, because it rewards studio-clean sound, not human delivery. Its extra
  letter error is one word: «بصدق» heard as «بصدقك».
- Guidance 1.5 is worse than 2.0 on every measure.
- No metric measures "sounds like AI"; the samples were sent to Yazan for listening.

## Listener verdict and follow-up

Of the 12 clips, Yazan found only one real-sounding: `A`, sentence 3 («يعني والله ما كنت ناوي أحكيلك، بس هيك صار، شو
بدي أعمل»). It is 12 words, so the tool did not force its timing: the model estimated it from Omar's reference
clip. It is also the only sentence written the way people talk, with «يعني» and «هيك».

The model's own estimate is close to his pace (sentence 1: 3.9 s estimated, 3.6 s at his pace, 7.0 s forced).

`spoken.py` tests both together: five sentences written as speech, model timing, otherwise setting A. Letter error
is 0 on three sentences and 0.023/0.028 on the other two («ما نمت منيح» → «ما نمتمني», «ومنحكي» → «وما نحكي»).
The clips were sent for listening.
