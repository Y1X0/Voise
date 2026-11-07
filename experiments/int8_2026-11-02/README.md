# int8 dynamic quantisation of Lahgtna on CPU: rejected (2026-11-02)

**Setup.** Both runs used the same voice prompt (your own recordings, reference chosen by `select_reference`), the
same three sentences, seed 0 and 32 steps. The int8 run applied `torch.ao.quantization.quantize_dynamic` to every
`nn.Linear` of the Qwen3 backbone; the audio codec and heads stayed float32.

**Measures.**
- **s / take:** seconds to generate one take.
- **CER:** letter error rate from the dialectal Whisper.
- **similarity:** ECAPA cosine to the reference.
- **naturalness:** UTMOS22, 1–5.

| | s / take | CER | similarity | naturalness |
|---|---|---|---|---|
| float32 | 420 | 0.219* | 0.718 | 3.06 |
| int8 (dynamic, all Linear) | **115** (3.6× faster) | 1.105 | 0.294 | 1.67 |

\* One float32 take was misheard (CER 0.66); its other two takes were perfect. The tool's best-of-3 selection
exists for takes like that.

**Decision: not adopted.** int8 is 3.6× faster, but the speech is no longer the sentence: the letter error rate is
above 1, and both voice similarity and naturalness collapse. A gentler scheme might be worth a separate test,
for example quantising only the MLP layers, or bfloat16 on CPUs that support it.
