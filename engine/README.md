---
title: voise-engine
emoji: 🎙
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Voise voice engine

A private-use voice engine for the owner's Telegram bot. It runs on a free Hugging Face Space (CPU), and every
request needs the `X-Engine-Secret` header. It speaks text, or re-speaks a voice note, in the voice of a reference
recording. Use it only with voices whose owners have consented.

- `POST /tts`: form fields `text`, `ref` (audio), optional `ref_text` and `language` (`apc` by default).
- `POST /s2s`: form fields `source` (voice note) and `ref` (audio). The note is transcribed with
  Whisper-large-v3-turbo, then the transcript is spoken in the reference voice.
- `GET /health`

**Models.** These are baked into the image at build time (see `fetch_models.py`):

| Model | Licence | Allowed use |
|---|---|---|
| Lahgtna-OmniVoice | None stated | Private use only, until its author permits more |
| Higgs Audio v2 tokenizer | Boson licence | As that licence allows |
| Whisper-large-v3-turbo | MIT | Unrestricted |

The code depends on OmniVoice, which is Apache-2.0.

**Expected speed.** On the free CPU tier a sentence takes a few minutes.
