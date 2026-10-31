# Restoration of WhatsApp voice notes before zero-shot cloning (2026-10-27)

**Question.** Does cleaning or restoring low-quality voice notes keep each speaker's identity? And does it make
one-clip cloning with Lahgtna-OmniVoice sound more like the person?

**Data.** We used 100 WhatsApp voice notes from five of Yazan's friends: Khaled, Muhannad, Mohammad, Majd and
Shahd. Yazan states they recorded and sent them for his graduation project. The clips total 9.8 minutes. Speakers
were grouped with WavLM-SV clustering, and Yazan named each group. Of the 100 clips, 88 have at least 2 s of
speech. None of the audio is in the repository; it stays under `runs/`, which is git-ignored.

**Conditions.**
- `raw`: the voice note as received.
- `sepformer`: noise removed with speechbrain/sepformer-dns4-16k-enhancement (Apache-2.0).
- `sidon`: speech restoration with Sidon (sarulab-speech/sidon-v0.1, via soniqo/Sidon-ONNX fp32, MIT). Sidon
  removes noise and reverberation and restores bandwidth. It outputs 48 kHz, which we resampled to 16 kHz.

**Metric.** Cosine similarity between ECAPA speaker embeddings (speechbrain/spkrec-ecapa-voxceleb). The speech
regions are found once, on the raw clip, and reused for every condition.

## 1. Identity (`identity.py`)

| | raw | sepformer | sidon |
|---|---|---|---|
| same speaker, mean cosine ↑ | 0.412 | 0.421 | **0.443** |
| different speakers, mean cosine ↓ | **0.148** | 0.155 | 0.220 |
| gap (same − different) ↑ | 0.264 | **0.266** | 0.223 |
| EER, same vs different ↓ | 0.211 | **0.191** | 0.237 |
| similarity to the clip's own raw version | 1.000 | 0.971 | 0.838 |

Sidon makes clips of the same person more alike. It makes different people more alike by a larger margin, so
speakers become harder to tell apart: EER rises from 0.211 to 0.237. This means part of the restored timbre comes
from the restorer, not from the speaker. Denoising alone barely changes the voice, which stays at 0.971 similarity
to its raw version, and it slightly improves speaker separation (EER 0.191).

## 2. One-clip cloning (`clone_eval.py`)

Each speaker was cloned from the same reference clip, cleaned per condition. Each clone is scored as its mean
cosine to that speaker's held-out raw clips.

| speaker (held-out clips) | raw | sepformer | sidon |
|---|---|---|---|
| Khaled (11) | 0.224 | 0.249 | 0.248 |
| Muhannad (7) | 0.511 | 0.496 | 0.509 |
| Mohammad (5) | 0.422 | 0.367 | 0.392 |
| Majd (2) | 0.489 | 0.471 | 0.476 |
| Shahd (1) | 0.428 | 0.333 | 0.471 |
| mean / weighted by held-out clips | 0.415 / 0.368 | 0.383 / 0.358 | 0.419 / 0.372 |

Whisper large-v3-turbo transcribed every clone correctly, apart from one-letter slips (خلص → خلق or خالص).

**Reading.** Cleaning the reference produces no consistent gain in one-clip cloning similarity. The differences
are within the noise of one sample per condition, and Majd and Shahd have only one or two held-out clips. The
restorer's homogenising effect (section 1) offsets its cleaner input.

## Conclusions

1. Off-the-shelf restoration does not by itself make clones sound like the person, and the strongest restorer
   partly erases speaker identity on these Arabic voice notes.
2. The limit is the reference itself: one short voice note per person. Same-speaker raw similarity is only
   0.36–0.62, compared with about 0.15 between different people.
3. Next steps:
   - Clean, longer recordings of a consenting speaker, used for the reference and for per-speaker adaptation.
   - As a research direction, an identity-preserving restorer. For example, fine-tune a Sidon-like model with a
     speaker-consistency loss, and judge it by the EER and gap in section 1.
