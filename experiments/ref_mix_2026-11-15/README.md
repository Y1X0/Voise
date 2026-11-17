# Diverse vs typical references (roadmap step 3), 2026-11-16

`ref_mix.py`: 16 voices, 3 held-out real clips each, three references of up to 19 s built from the person's other
clips, each voice on its chosen version, seed 0, one take, no VC. Run on a private Kaggle T4 (`kaggle_run.py`;
owner decision in `docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md`, R15; dataset and notebook deleted afterwards).

- **tool**: what Voise does today (`select_reference`: 7-12 s stretches ranked by noise, Whisper clarity and
  typicality, joined up to 19 s; it reached only ~15 s on most voices)
- **typical**: whole clips closest to the person's voice centre, up to 19 s (step 2's reference)
- **diverse**: most typical clip, then clips most unlike the chosen ones in pitch, pitch range, loudness and speaking
  rate, among the more typical half

| reference | mean closeness | best for | letter error | pitch range (clone / real) |
|---|---|---|---|---|
| tool    | 92.8%  | 1/16  | 3.4% | 1.01 |
| typical | 102.2% | 12/16 | 3.7% | 0.88 |
| diverse | 99.4%  | 3/16  | 3.4% | 0.93 |

Per voice, closeness / letter error % (tool, typical, diverse):
p12 86/3.2 97/3.0 94/2.3; p140 84/4.8 94/1.2 89/1.4; p152 107/0.4 99/1.0 106/3.2; p30 52/2.9 107/2.8 105/0.4;
p48 110/3.4 113/1.6 108/0.9; abood 102/1.0 110/3.6 104/2.7; fakhri 107/2.8 115/4.4 108/0.9; p106 85/3.1 96/6.4 92/11.3;
p43 96/1.9 101/3.8 90/1.5; p51 107/4.3 115/9.8 118/4.7; p64 101/10.5 107/7.6 101/10.1; p70 81/5.7 97/2.8 90/4.6;
p72 102/4.3 105/3.0 102/4.9; p85 86/4.1 91/5.2 91/5.2; p95 87/0.0 99/2.1 99/0.0; omar 92/1.9 91/1.7 94/0.3.

Replicate: the same script on CPU (here) for the first 5 voices agrees in direction: tool < typical on 4 of 5, and
p30's tool reference is broken in both (38% CPU, 52% GPU vs ~105% for the other two). One seed per text, so a single
voice's number moves by several points between runs (p140 typical 87 vs 94); the means and the p30 gap are robust.

Verdict:
- **Typical beats the tool's reference** on 14 of 16 voices (+9.4 points on average); it loses on p152 and omar by
  1-8 points. Part of the gain may be length (19 s vs ~15 s), since the tool cannot reach 19 s with 7-12 s pieces.
- **Diverse is not better than typical** (better on 3 of 16), and it does not make the clone livelier on average.
  Not adopted.
- **Cost**: typical reads slightly flatter (pitch range 0.88 vs 1.01) and its letter error is 0.3 points higher on
  average, mostly p51 (9.8%), p106 (6.4%) and abood (3.6%). By roadmap rule 2 it cannot replace the tool's choice
  as it is: it needs a letter-error check per clip (Whisper clarity, which the tool already computes) before it
  goes in.
- **p30**: the tool's reference is the cause of p30's weak clones (52%, not the model). Typical fixes it (107%).
