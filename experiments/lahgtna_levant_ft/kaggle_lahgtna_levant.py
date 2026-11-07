"""Kaggle GPU: continue training Lahgtna-OmniVoice on Levantine / Jordanian speech, so one-clip cloning in those
dialects improves for every speaker (not one person's adapter).

Data (downloaded here, never re-uploaded; the resulting weights are for private research only, since the Lahgtna
weights state no licence and SYRA does not cover third-party rights in its source recordings):
  * SYRA - Overtech/syrian-arabic-speech-dataset (SYRA Research Dataset License: research / education /
    non-commercial). Syrian podcast speech, grouped by its `source` field.
  * DataHive - datahiveai/arabic-multidialect-emotional-speech-demo (CC BY-NC 4.0). Acted emotional speech, incl.
    Jordanian; oversampled so emotion and Jordanian dialect are not drowned out.
Held out: one Jordanian DataHive speaker and one SYRA source, never trained on; masked-token loss on them is
reported before and after training. The real evaluation (one-clip cloning of consented friends' voices) runs
outside Kaggle on the downloaded weights, so no private audio ever reaches Kaggle.

Recipe: OmniVoice's own sample processor (random prompt prefix, random mask ratio, loss on masked audio tokens);
all weights trained, fp16 autocast with fp32 master weights, AdamW, warm-up then cosine decay, time-bounded.
Output: /kaggle/working/lahgtna-levant/ = the original model folder with model.safetensors replaced.
"""
import glob, io, json, math, os, random, shutil, subprocess, sys, time
MODE = "full"
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
def sh(cmd): log("$", cmd); subprocess.run(cmd, shell=True, check=True)

CFG = dict(smoke=dict(syra_hours=0.3, datahive_rows=40, steps=30, train_seconds=600, accum=2, eval_items=8),
           full=dict(syra_hours=30.0, datahive_rows=10**9, steps=8000, train_seconds=int(6.0 * 3600), accum=8,
                     eval_items=60))[MODE]
WORK, OUT = "/tmp/lv", "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
SEED = 0
random.seed(SEED)

sh("pip install -q --no-deps omnivoice==0.2.1")
sh("pip install -q transformers==5.18.0 accelerate soundfile librosa pyarrow huggingface_hub webdataset tensorboardx pydub")

import numpy as np, soundfile as sf, torch, librosa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import save_file

MODEL = snapshot_download("oddadmix/lahgtna-omnivoice-v2", revision="55c38a613e31", local_dir=f"{WORK}/lahgtna",
                          allow_patterns=["*.json", "*.safetensors", "*.jinja"])
snapshot_download("eustlb/higgs-audio-v2-tokenizer", revision="528e871c2a26", local_dir=f"{MODEL}/audio_tokenizer",
                  allow_patterns=["*.json", "*.safetensors"])
dev = "cuda"
TSR = 24000

# ---------------------------------------------------------------- data
def decode(audio_cell):
    b = audio_cell["bytes"] if isinstance(audio_cell, dict) else audio_cell
    y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=TSR) if sr != TSR else y

items = []          # (group, text, wav24k, weight, heldout)
api = HfApi()
syra_files = sorted(f for f in api.list_repo_files("Overtech/syrian-arabic-speech-dataset", repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet"))
random.Random(SEED).shuffle(syra_files)
secs = 0.0
for f in syra_files:
    p = hf_hub_download("Overtech/syrian-arabic-speech-dataset", f, repo_type="dataset", local_dir=f"{WORK}/syra")
    t = pq.read_table(p, columns=["audio", "transcript_text", "duration", "source"]).to_pylist()
    os.remove(p)
    for r in t:
        d = float(r["duration"] or 0)
        if not 2.0 <= d <= 14.0 or not (r["transcript_text"] or "").strip():
            continue
        items.append(("syra:" + (r["source"] or "?"), r["transcript_text"].strip(), r["audio"], 1.0))
        secs += d
        if secs >= CFG["syra_hours"] * 3600:
            break
    log(f"SYRA {f}: {secs / 3600:.1f} h so far")
    if secs >= CFG["syra_hours"] * 3600:
        break

dh_files = [f for f in api.list_repo_files("datahiveai/arabic-multidialect-emotional-speech-demo", repo_type="dataset")
            if f.endswith(".parquet")]
dh = []
for f in dh_files:
    p = hf_hub_download("datahiveai/arabic-multidialect-emotional-speech-demo", f, repo_type="dataset", local_dir=f"{WORK}/dh")
    dh += pq.read_table(p, columns=["audio", "text", "dialect", "recording_style", "speaker_id", "qa_tier"]).to_pylist()
random.Random(SEED).shuffle(dh)
dh = dh[: CFG["datahive_rows"]]
jo_speakers = sorted({r["speaker_id"] for r in dh if (r["dialect"] or "").lower().startswith("jordan")})
HELD_DH = jo_speakers[0] if jo_speakers else None
for r in dh:
    if (r["text"] or "").strip():
        w = 6.0 if (r["dialect"] or "").lower().startswith("jordan") else 3.0
        items.append(("dh:" + r["speaker_id"], r["text"].strip(), r["audio"], w))
groups = sorted({g for g, *_ in items if g.startswith("syra:")})
HELD_SYRA = groups[-1] if len(groups) > 1 else None
if HELD_SYRA is None:          # one SYRA source only: hold out a random 3% of its clips instead
    idx = [i for i, it in enumerate(items) if it[0].startswith("syra:")]
    for i in random.Random(SEED).sample(idx, max(1, len(idx) // 33)):
        items[i] = ("syra-heldout",) + items[i][1:]
    HELD_SYRA = "syra-heldout"
if HELD_DH is None:            # no Jordanian speaker in this subset: hold out one DataHive speaker
    HELD_DH = sorted({r["speaker_id"] for r in dh})[0] if dh else None
held = {g for g in (f"dh:{HELD_DH}" if HELD_DH else None, HELD_SYRA) if g}
log(f"{len(items)} items; groups {len(set(g for g, *_ in items))}; held out {held}")

# ---------------------------------------------------------------- audio tokens
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
fe = AutoFeatureExtractor.from_pretrained(f"{MODEL}/audio_tokenizer")
codec = HiggsAudioV2TokenizerModel.from_pretrained(f"{MODEL}/audio_tokenizer").to(dev).eval()
train, evals = [], []
for i, (g, text, audio, w) in enumerate(items):
    try:
        y = decode(audio)
    except Exception as e:
        log("skip", g, e); continue
    with torch.inference_mode():
        codes = codec.encode(fe(raw_audio=y, sampling_rate=TSR, return_tensors="pt")["input_values"].to(dev)).audio_codes.squeeze(0).cpu()
    if codes.shape[1] > 14 * 25 + 10:          # longer than ~14 s at 25 frames/s: skip (memory on a T4)
        continue
    s = {"audio_tokens": codes, "label": {"text": text, "language_id": "apc"}, "w": w, "g": g}
    (evals if g in held else train).append(s)
    if i % 500 == 0:
        log(f"tokenised {i}/{len(items)}")
del codec, items
torch.cuda.empty_cache()
log(f"train {len(train)}  eval {len(evals)}")
evals = evals[: CFG["eval_items"]]

# ---------------------------------------------------------------- model
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.models.omnivoice import OmniVoice
model = OmniVoice.from_pretrained(MODEL, device_map=dev, dtype=torch.float32)
proc = OmniVoiceSampleProcessor(text_tokenizer=model.text_tokenizer, num_channels=model.config.num_audio_codebook,
                                audio_mask_id=model.config.audio_mask_id, prompt_ratio_range=(0.0, 0.3),
                                mask_ratio_range=(0.0, 1.0), drop_cond_ratio=0.1, language_ratio=0.8,
                                use_pinyin_ratio=0.0, instruct_ratio=0.0, only_instruct_ratio=0.0)

def step_loss(s):
    b = proc(s)
    with torch.autocast("cuda", dtype=torch.float16):
        o = model(input_ids=b["input_ids"][None].to(dev), audio_mask=b["audio_mask"][None].to(dev),
                  labels=b["labels"][None].to(dev))
    return o.loss if hasattr(o, "loss") else o["loss"]

@torch.no_grad()
def eval_loss():
    model.eval()
    rs = random.getstate(); tr = torch.random.get_rng_state()
    random.seed(123); torch.manual_seed(123)
    ls = [float(step_loss(s)) for s in evals for _ in range(2)]
    random.setstate(rs); torch.random.set_rng_state(tr)
    model.train()
    return round(float(np.mean(ls)), 4)

res = {"mode": MODE, "train_items": len(train), "eval_items": len(evals), "held_out": sorted(held)}
res["eval_loss_before"] = eval_loss(); log("eval loss before", res["eval_loss_before"])

model.llm.gradient_checkpointing_enable()
model.llm.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(True)
for p in model.audio_tokenizer.parameters() if model.audio_tokenizer is not None else []:
    p.requires_grad_(False)
params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("audio_tokenizer.")]
opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
total = CFG["steps"]
warm = min(300, max(1, total // 10))
lr_at = lambda k: 1e-5 * min(1.0, k / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, k / max(1, total)))) * 0.9 + 0.1)
weights = np.array([s["w"] for s in train]); weights /= weights.sum()
model.train()
t_start, losses, k = time.time(), [], 0
while k < total and time.time() - t_start < CFG["train_seconds"]:
    for g_ in opt.param_groups:
        g_["lr"] = lr_at(k)
    oom = False
    for _ in range(CFG["accum"]):
        try:
            loss = step_loss(train[int(np.random.choice(len(train), p=weights))]) / CFG["accum"]
            scaler.scale(loss).backward()
            losses.append(float(loss) * CFG["accum"])
        except torch.OutOfMemoryError:
            oom = True
            break
    if oom:                                     # drop this update rather than crash the run
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        res["oom_skips"] = res.get("oom_skips", 0) + 1
        continue
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    k += 1
    if k % 100 == 0 or MODE == "smoke":
        log(f"step {k} loss {np.mean(losses[-100 * CFG['accum']:]):.3f} lr {lr_at(k):.2e}")
    if k % 1500 == 0:
        log("eval loss", eval_loss())
res["steps"] = k
res["train_loss_first_last"] = [round(float(np.mean(losses[:200])), 4), round(float(np.mean(losses[-200:])), 4)]
res["eval_loss_after"] = eval_loss(); log("eval loss after", res["eval_loss_after"])

# ---------------------------------------------------------------- save: the original folder, new model.safetensors
dst = f"{OUT}/lahgtna-levant"
# skip only the top-level weights we replace; audio_tokenizer/model.safetensors (the frozen codec) must be kept
shutil.copytree(MODEL, dst, ignore=lambda d, names: [n for n in names if n == ".cache" or (d == MODEL and n == "model.safetensors")],
                dirs_exist_ok=True)
sd = {n: t.detach().float().cpu().contiguous() for n, t in model.state_dict().items() if not n.startswith("audio_tokenizer.")}
save_file(sd, f"{dst}/model.safetensors", metadata={"format": "pt"})
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
log("DONE", res)
