# Content teacher decision (B3)

## Decision: **TEACHER_RESOLVED — Whisper-small encoder (MIT), offline only**

The decision needs the owner's confirmation (**U2**). The choice between the commercially
safe candidates is made by a pre-registered ablation on VALID (§4).

**mHuBERT-147 is rejected.** Its official Hugging Face card states
`license: cc-by-nc-sa-4.0`, which makes it **NON_COMMERCIAL**.

**`TEACHER_LICENSE_BLOCKED` does not apply.** Three teachers have E1 evidence of a
commercial-permissive licence (MIT or Apache-2.0) and cover Arabic.

## 1. What the teacher is used for (and what it is not)

**Stage 1 — unit targets, precomputed OFFLINE:**
* The teacher encodes each training utterance.
* An intermediate layer goes through k-means with K = 500, producing uint16 units at 20 ms.
  StreamAnon's 10 ms frames repeat each unit twice.
* Implementation: `training/scripts/compute_teacher_units.py`, about 0.5 GB of units for 1 500 h.
* After this step **the teacher is not needed by training, and never by the product.**

**Stage 3 — content-output loss.** This loss uses a **frozen copy of the stage-1 student
encoder**, not the external teacher. This avoids an online teacher in the training loop.
Whisper's fixed 30 s window would make an online loss on 2 s segments about 15× wasteful.

**Runtime and internet:**
* none of the candidates is ever shipped;
* the Android model contains only the StreamAnon student;
* there is no runtime internet access and no teacher at inference time.

## 2. Candidates (official sources, read directly)

| Teacher | Official checkpoint | Licence (E1) | Commercial | Size | Languages (incl. Arabic?) | Frame rate | Compute for 1 500 h units | Runtime internet | Training-only | Units precomputable → removed from product |
|---|---|---|---|---|---|---|---|---|---|---|
| **Whisper small (multilingual) encoder** | `openai/whisper` (`small`); HF `openai/whisper-small` | MIT: "Whisper's code and model weights are released under the MIT License" | **CWC** (MIT notice) | 244 M params total, about 88 M in the encoder; encoder FP32 ONNX 0.41 GB | ~99 incl. **ar** (card) | 20 ms (1 500 frames / 30 s) | **Measured** (§3): RTF 0.03 per 30 s window on 4 CPU threads. Padding short utterances to 30 s costs 2–3×, so about 90–135 CPU-hours, or about 1–2 GPU-hours. | no | yes | **yes** |
| XLS-R 300M | `facebook/wav2vec2-xls-r-300m` (fairseq) | HF card `apache-2.0`; fairseq MIT incl. pretrained models | **CWC** | 300 M; 1.27 GB | 128 incl. ar | 20 ms | similar order (no 30 s padding) | no | yes | yes |
| w2v-BERT 2.0 | `facebook/w2v-bert-2.0` | HF card `license: mit` | **CWC** | 600 M | 143 incl. **ar, ary, arz** | 20 ms | about 2× XLS-R | no | yes | yes |
| HuBERT base (LS960) | fairseq / `facebook/hubert-base-ls960` | fairseq MIT; card apache-2.0 | **CWC** | 95 M | **English only** | 20 ms | small | no | yes | yes (English ablation only) |
| WavLM Base+ | `microsoft/wavlm-base-plus` | card states no licence; repo LICENSE MIT, applicability to the weights unstated | **LICENSE_NOT_VERIFIED** | 95 M | English | 20 ms | — | — | — | not used |
| mHuBERT-147 | `utter-project/mHuBERT-147` | `cc-by-nc-sa-4.0` | **NON_COMMERCIAL** | 95 M | 147 incl. ar | 20 ms | — | — | — | **rejected** |

**Compatibility with StreamAnon.** All candidates are compatible:
* units at 20 ms, which maps directly to 2 student frames;
* k-means discrete targets;
* the student's causal encoder learns them through `unit_ce`.

Units carry less speaker information when k-means is fitted on per-speaker mean-normalised
features. `compute_teacher_units.py` documents this, and speaker purity is reported.

## 3. Measurement done here (CPU, no GPU)

The Whisper-small encoder was measured with the sherpa-onnx `small.en` export, which has
the same encoder architecture and size as multilingual `small`. Setup: ONNX Runtime CPU,
4 threads, `ORT_DISABLE_TELEMETRY=1`.

* Input `[1, 80, 3000]` produces 1 500 frames, i.e. 20 ms per frame.
* **0.91 s per 30 s window, so RTF 0.03.**
* This export outputs cross-attention K/V. Real unit extraction uses the official PyTorch
  checkpoint's intermediate hidden states, which cost the same compute.

Unit extraction is therefore feasible **without a GPU**. It is the one heavy preprocessing
step, and it is not training.

## 4. Why Whisper is first choice, and how the final pick is made

**Why Whisper first:**
* Its licence evidence is the clearest: one MIT statement covering code **and** weights,
  in the official README.
* It covers Arabic.
* Its supervised ASR training makes its intermediate features comparatively
  speaker-invariant, which is what an anonymizer's content path needs.

**Risks:**
* Whisper encoder k-means units are less established than SSL (HuBERT-style) units.
* The 30 s window wastes compute (offline only).
* Whisper's training data is undisclosed. Only the weights' licence is MIT. This is a
  residual legal question for counsel, not a licence restriction.

**Pre-registered ablation** (VALID only; CPU-feasible on about 10 h):
* Candidates: Whisper-small layers {6, 8, 10}, XLS-R 300M layers {12, 15, 18}, and
  w2v-BERT 2.0 layers {12, 16}.
* Each candidate uses K = 500 k-means fitted on speaker-normalised features.
* Metrics:
  1. phonetic purity: a linear CTC probe from unit one-hot to characters (`ctc_targets`), CER;
  2. speaker leakage: speaker-ID accuracy of a linear probe on unit histograms;
  3. Arabic and English separately.
* **Rule, fixed now:** choose the lowest CER among candidates whose speaker-probe accuracy is
  ≤ 1.2× the minimum. Ties go to Whisper.
* The result is recorded here, and the teacher is fixed **before** stage 1. TEST data is
  never used.
