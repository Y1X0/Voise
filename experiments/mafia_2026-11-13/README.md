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
