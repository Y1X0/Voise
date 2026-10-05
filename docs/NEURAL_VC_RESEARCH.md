# Neural voice conversion: feasibility for real-time Android anonymization

Status: **research only, nothing implemented.** Figures for third-party systems
are as reported by their authors or typical published orders of magnitude. They
are **NOT VERIFIED** in this project. Figures for our DSP are measured (see
`ANONYMIZATION_EVALUATION.md`).

Goal restated: turn the user's voice into a *different, synthetic* voice while
keeping the words, in real time, offline, on Android. **Not** imitating any
real person. Voice cloning of a real target speaker is out of scope.

## 1. Candidate families

| Approach | Latency | CPU | RAM | Model size | Offline | Android feasibility | Naturalness | Anonymization potential | Privacy | Licensing |
|---|---|---|---|---|---|---|---|---|---|---|
| **Traditional DSP (this app: PSOLA pitch + formant + spectral)** | 32 ms algorithmic (measured) | ~1.5 % of one desktop core; ~2 % on emulator (measured) | < 20 MB native heap (emulator) | none | yes | **runs today** (emulator-verified) | good for moderate shifts; degrades at large shifts | **limited** (measured: ignorant-attacker linkage reduced, lazy-informed linkage largely kept) | on-device | own code |
| McAdams-coefficient LPC warping (VoicePrivacy signal-processing baseline) | ~20–30 ms frame-based | very low | tiny | none | yes | easy | moderate (some "buzzy" quality) | similar class to ours; published results show weak protection against an informed attacker | on-device | simple algorithm |
| Classical statistical VC (GMM / parallel data) | 10–50 ms | low–moderate | small | small | yes | possible | often muffled/"vocoded" | moderate, but needs parallel training data for a *target* voice | on-device | depends on data |
| x-vector + content features + neural vocoder (VoicePrivacy "B1"-style: ASR bottleneck features, pseudo-speaker x-vector, NSF/HiFi-GAN) | batch / utterance-level (ASR features) | high (TDNN ASR + vocoder) | hundreds of MB | 100+ MB | yes | **not real-time on phones** as designed | good | strong against an ignorant attacker, partial against an informed one (published) | on-device possible | research code; dataset licences vary |
| kNN-VC (WavLM features + nearest-neighbour matching + HiFi-GAN) | non-streaming (utterance) | very high (WavLM-Large ~300 M params) | > 1 GB | > 1 GB | yes | **no** | good | strong (replaces features with another speaker's), but needs reference audio of a target voice | on-device impossible today | MIT code; WavLM licence |
| RVC (HuBERT/ContentVec + F0 + NSF-HiFi-GAN) | ~100–300 ms in practice | GPU-class | 0.5–1 GB | ~50–200 MB | yes | **not realistic** on phone CPUs in real time | good when trained | strong, but ecosystem is built around cloning real people (ethics) | on-device possible | mixed |
| Streaming lightweight VC (e.g. LLVC 2023, StreamVC 2024) | ~20–70 ms reported | reported real-time on a single CPU core / on-device | tens of MB | ~5–30 MB | yes | **plausible**, not demonstrated here | reported good | strong *if* the target is a voice that is not the user's | on-device | LLVC: code+weights released (MIT), but the released target voice is a real recorded speaker, so retraining toward a synthetic pseudo-speaker would be required; StreamVC: no public weights |
| Neural vocoder resynthesis of modified DSP features (e.g. LPCNet / small HiFi-GAN from cepstral envelope + F0) | ~10–30 ms | LPCNet ~3 GFLOPS reported (real-time on phones) | small | ~1–3 MB (LPCNet) | yes | plausible | good–moderate (vocoder artifacts on noisy input) | removes excitation fine-structure cues; envelope trajectories (main identity cue) remain unless also mapped | on-device | BSD (LPCNet) |

## 2. What the data says about where identity remains

From `ANONYMIZATION_EVALUATION.md` (15 real speakers):

* Pitch/formant/spectral changes lower original→processed similarity. The
  *lazy-informed* attacker (who also processes the enrolment) still links
  processed recordings of the same speaker well. Any **deterministic per-user
  transform** keeps the speaker's relative identity structure: it moves
  everyone, but it keeps speakers apart from each other.
* Per-session randomisation and stronger envelope changes help somewhat, at a
  cost in naturalness. The exact numbers are in the evaluation document.

INTERPRETATION: really removing identity needs a method that **replaces** the
speaker-dependent part instead of *transforming* it: content features plus a
synthetic target voice (VC), or a pseudo-speaker. That is the case for neural VC
in principle.

## 3. Architecture that could realistically run on Android (if pursued later)

```
mic ─▶ 16 kHz frames (10–20 ms)
     ─▶ small streaming content encoder (causal conv, distilled from a self-supervised model; ~5–10 M params, int8)
     ─▶ F0 tracker (existing YIN) ─▶ F0 re-mapped to the pseudo-speaker's range
     ─▶ causal decoder/vocoder conditioned on ONE fixed synthetic pseudo-speaker embedding
          (mean of many training speakers + random offset, verified not close to any real speaker)
     ─▶ existing limiter ─▶ out
runtime: LiteRT (TFLite) or ONNX Runtime Mobile, XNNPACK on CPU; target ≤ 40 ms extra latency, ≤ 25 % of one big core
```

Requirements before writing any code:

1. A training corpus with licences that permit this use. Speakers are used only
   to *average* into a pseudo-speaker, never as a target voice.
2. An offline prototype that beats the best DSP configuration here on **both**
   ignorant and lazy-informed EER (same corpus, same embedding model, plus a
   second embedding model), with relative WER and DNSMOS no worse than Balanced.
3. A device benchmark: latency, CPU, thermals and battery for 30 minutes on a
   mid-range phone.
4. A licence review of every model and dataset.

## 4. Decision for this phase

Neural VC is **not added**. No model available in this environment meets all of:

* real time on Android CPUs
* public weights
* a synthetic (non-real-person) target voice
* a permissive licence

Training one is a separate project with its own data and compute requirements.
The current evidence only says that DSP transforms leave linkability. It does
not show that a specific neural model would do better *on Android in real
time*. See the final decision in `ANONYMIZATION_EVALUATION.md`.
