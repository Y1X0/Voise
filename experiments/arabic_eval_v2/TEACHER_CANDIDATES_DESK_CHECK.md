# Alternative-teacher desk check (2026-10-11): one candidate eligible — **OpenVoice V2**

## Scope and method

- This is a read-only desk check. Nothing was run, and there was no audio download, no GPU and no training.
- Sources are only the public licence files, READMEs and Hugging Face cards, read on this date.
- The candidates are the three proposed after kNN-VC failed (`TEACHER_GATE_RESULT.md`).
- Eligibility uses the unchanged T4 of `TEACHER_GATE.md`: teacher code **and** weights permissive, target synthetic.

## Findings (OBSERVED)

| Item | OpenVoice V2 | FreeVC | Seed-VC |
|---|---|---|---|
| Official repo | `myshell-ai/OpenVoice` | `OlaWod/FreeVC` | `Plachtaa/seed-vc` (`lsgzt/seed-vc` is a copy, also GPL-3.0) |
| Code licence | MIT (LICENSE; README: "V2 and V1 are released under MIT License. Free for commercial use") | MIT | **GPL-3.0** |
| Weights: location | HF `myshell-ai/OpenVoiceV2`: `converter/checkpoint.pth` (131.3 MB) and `config.json`; also an S3 zip | **OneDrive link only** (`1drv.ms`) | HF `Plachta/Seed-VC` |
| Weights: licence | HF card `license: mit` | **No statement** separate from the code | HF card **`license: gpl-3.0`**, so this is determined |
| Upstream models | No SSL model in the converter path | WavLM-Large (microsoft/unilm; the HF card has no licence tag). The speaker encoder is not in the repo tree and has no download link | v2: `facebook/hubert-large-ll60k` (apache-2.0), `funasr/campplus` (apache-2.0), `nvidia/bigvgan_v2_22khz_80band_256x` (mit), whisper-small tokenizer (MIT), ASTRAL-quantization (**GPL-3.0**) |
| Python deps of concern | `whisper-timestamped` is **AGPL-3.0**. It is imported only in `openvoice/se_extractor.py` (VAD splitting) and is avoidable by calling `ToneColorConverter.extract_se` directly. `wavmark` (MIT) adds a watermark and can be disabled. `faster-whisper` is MIT | – | (not audited further, because the licence already disqualifies it) |
| Arabic | The 6 "native" languages refer to TTS base speakers. Per README/USAGE, converter input can be "Any Language" and the reference language need not be in training. Arabic quality is **unmeasured** | Trained on VCTK (English); Arabic unmeasured | Arabic unmeasured |
| Synthetic fixed target | Yes. Target tone embedding = `extract_se` over the 60 WORLD pseudo-speaker renders (seed 4242) already used for kNN-VC, giving one fixed embedding with no real person | Yes in principle: a speaker embedding from the same renders | Yes in principle: the same renders as reference |
| Latency (for information; the teacher is offline, so this is not a gate) | not stated | not stated | README: algorithmic ~300 ms + device ~100 ms. The 1.27–4.07 s figure cited earlier is not in the official README |

## Eligibility under unchanged T4

| Candidate | Eligible | Reason |
|---|---|---|
| **OpenVoice V2** | **yes** | Code and weights MIT. The only copyleft dependency (AGPL `whisper-timestamped`) is optional and must be excluded from the test script |
| FreeVC | no (unverifiable) | Weights have no licence statement and are hosted only on OneDrive; the speaker-encoder source is not documented. This is the same status as B3 in `TEACHER_GATE.md` |
| Seed-VC | no | Code and weights are GPL-3.0, and T4 requires permissive. Whether GPL would reach an LLVC student trained on its outputs is legally unclear; that question is not resolved here |

## Interpretation

- OpenVoice V2 is the only candidate that clears T4 without changing any criterion.
- Its converter swaps "tone colour" but by design keeps the source rhythm and intonation. Higher intelligibility (T1) is therefore plausible.
- Weaker privacy (T2/T3) is a real risk, because prosody carries identity.
- None of this is known until measured.

## Limitations

- Licence reading is from the repositories' own statements and is not legal advice.
- The OpenVoice V2 training data is not identified in the files read.
- The licence of the CC BY SVQ pool does not settle privacy rights. Pool residue must be measured as before.
- No candidate's Arabic quality or privacy has been measured.

## Next step (needs approval; nothing run)

- Pre-register a teacher gate for OpenVoice V2 with T1–T4 exactly as in `TEACHER_GATE.md`, using the same frozen 225 clips, pipeline, bootstrap and pool.
- Pin the HF commit and the sha256 of `converter/checkpoint.pth`. Disable the watermark. Do not import `se_extractor`.
- If OpenVoice V2 fails, the search widens (e.g. other permissively licensed VC models). Thresholds are not relaxed.
