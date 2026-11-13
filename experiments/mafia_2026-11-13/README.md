# Learning from a multi-speaker recording: Mafia episode 1 (2026-11-13)

**Source.** A 51-minute episode of a Mafia game show (12 players and a host). Yazan, the show's assistant
producer, states that he asked the participants and they agreed to their voices being used to train Voise. The
consent record is `runs/consent/mafia1.json`. No audio is in the repository.

**Steps.**
1. `segment.py` splits the episode into 682 single-turn segments of 1–8 s (40 min of speech). Each segment gets an
   ECAPA speaker embedding and a dialect-Whisper transcript.
2. Speakers are grouped by average-linkage cosine clustering at distance 0.6.
3. Dialect filter. Two voices use clearly North African forms («باهي»، «شنو»، «علاش»، «وش سامسوش ڨلتك»، «كليش
   مكشفت»): 7.3 min, left out so the model's accent stays Jordanian. Within the kept voices, any segment with such
   a form is also dropped.
4. Naming. A name said by one voice and answered within 5 s by another points to the answering voice:
   - «عبود» and «عبد الله» both point to the same voice (3 answers each, never said by it).
   - «فخري» points to one voice (2 answers; it also speaks right after the line «يلا فخري»).
   - The other eight voices have too little evidence and stay unnamed (`mafia_p<group>`). A sample of each is kept
     so a person who withdraws can be found and removed.
5. Clips shorter than 2 s or far from their voice's centre (cosine < 0.55, likely overlapping speech) are dropped.
   This leaves 10 voices, 302 clips and 23.8 min.

## Result: generation 1 of the rebuilt pool (accepted)

Pool: the 10 Mafia voices (291 clips after the quality filter) plus Omar (161 clips). Training: 600 steps from
the base model (generation 2 was lost with the container). Same acceptance test as `voise_learn.py`.

| | before (base model) | after |
|---|---|---|
| held-out loss | 6.058 | **5.476** |
| unseen person's loss | 6.046 | **5.281** |
| letter error (6 sentences, 2 voices) | 3.4% | **1.3%** |
| UTMOS | 2.74 | **2.92** |

The unseen person's loss falls by 0.77, much more than generation 2's 0.14 (5.22 → 5.08), and letter error falls
by more than half.
