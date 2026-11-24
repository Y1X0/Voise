"""Fine-tune the kNN-VC prematched HiFi-GAN on the Arabic ClArTTS reader ("prematched" training, Baas et al. 2023):
each training utterance's WavLM features are replaced by the mean of their k nearest frames from the OTHER
utterances of the pool, and the vocoder learns to rebuild the real audio from those matched features, i.e. from
the same kind of input it receives at conversion time.

Selection: Whisper-small CER against the TRUE ClArTTS transcripts of held-out TEST utterances (never in the pool
or training), vocoded from features matched against the pool. The owner's recording is not used here.

  VOISE_DL=... python3 finetune_vocoder.py FEATS_DIR OUT_DIR [--steps N] [--train-shards 3]
"""
import argparse, glob, json, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F, torchaudio
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts"), os.path.join(REPO, "training")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen
from unit_vocoder_pilot import MelLoss, MEL_LAMBDA
from models.discriminators import Discriminators
from losses import losses as L

ap = argparse.ArgumentParser()
ap.add_argument("feats"); ap.add_argument("out")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--train-shards", type=int, default=3)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--seg", type=int, default=32, help="frames per training segment (20 ms each)")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--adv-from", type=int, default=500, help="add GAN + feature-matching terms to G from this step")
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
torch.set_num_threads(os.cpu_count())
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
SR, HOP = 16000, 320
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
def log(*x): print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

# ---------------------------------------------------------------- data
shards = [torch.load(p, weights_only=True) for p in sorted(glob.glob(os.path.join(a.feats, "train-*.pt")))]
pool = torch.cat([s["feats"] for s in shards])                                     # float16 [N, 1024]
utt = []                                                                           # (global frame start, end, audio int16)
base = 0
for si, s in enumerate(shards):
    fo, ao = s["feat_offsets"], s["audio_offsets"]
    for i in range(len(s["files"])):
        if si < a.train_shards:
            utt.append((base + int(fo[i]), base + int(fo[i + 1]), s["audio"][int(ao[i]):int(ao[i + 1])]))
    base += len(s["feats"])
log(f"pool {len(pool) * 0.02 / 60:.1f} min; training utterances {len(utt)} ({sum(e - s for s, e, _ in utt) * 0.02 / 60:.1f} min)")
pool_n = F.normalize(pool.float(), dim=-1).to(torch.float16)


def match(q, k, exclude=None, block=60000):
    """kNN (cosine) of query frames against the pool, excluding the pool range `exclude`."""
    qn = F.normalize(q.float(), dim=-1)
    bs = torch.full((len(q), k), -9.0); bi = torch.zeros((len(q), k), dtype=torch.long)
    for s in range(0, len(pool_n), block):
        sim = qn @ pool_n[s:s + block].float().T
        if exclude is not None:
            lo, hi = max(exclude[0] - s, 0), min(exclude[1] - s, sim.shape[1])
            if lo < hi: sim[:, lo:hi] = -9.0
        ts, ti = sim.topk(k, dim=-1)
        cs, ci = torch.cat([bs, ts], 1), torch.cat([bi, ti + s], 1)
        bs, j = cs.topk(k, dim=-1); bi = ci.gather(1, j)
    return pool[bi].float().mean(1)


cache = os.path.join(a.out, f"prematched_k{a.k}_{a.train_shards}shards.pt")
if os.path.exists(cache):
    pm = torch.load(cache, weights_only=True)
else:
    pm = []
    for i, (s, e, _) in enumerate(utt):
        pm.append(match(pool[s:e], a.k, exclude=(s, e)).to(torch.float16))
        if i % 200 == 0: log(f"prematch {i}/{len(utt)}")
    torch.save(pm, cache)
log("prematched features ready")

# held-out selection set: ClArTTS TEST utterances with true transcripts
import pyarrow.parquet as pq
tb = pq.read_table(f"{HF}/clartts/data/test-00000-of-00001.parquet", columns=["file", "text", "audio", "sampling_rate"]).to_pydict()
sel = np.argsort(tb["file"])[:12]
wl = load_wavlm()
val = []
for i in sel:
    y = torchaudio.functional.resample(torch.from_numpy(np.asarray(tb["audio"][i], np.float32)), int(tb["sampling_rate"][i]), SR).numpy()
    val.append((match(wavlm_feats(wl, y), a.k), tb["text"][i]))
del wl, tb
log(f"validation: {len(val)} test utterances, {sum(len(v[0]) for v in val) * 0.02:.0f} s")

from transformers import WhisperForConditionalGeneration, WhisperProcessor
proc = WhisperProcessor.from_pretrained(f"{HF}/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained(f"{HF}/whisper-small").eval()
def norm(t):
    t = re.sub(r"[ً-ْـ]", "", t); t = re.sub(r"[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w\s]", " ", t).split()
def ed(x, y):
    d = list(range(len(y) + 1))
    for i in range(1, len(x) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(y) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x[i - 1] != y[j - 1]))
    return d[-1]
@torch.no_grad()
def val_cer(gen):
    gen.eval(); e = n = we = wn = 0
    for f, txt in val:
        y = gen(f[None])[0].numpy()
        ids = asr.generate(proc(y, sampling_rate=SR, return_tensors="pt").input_features, language="ar", task="transcribe",
                           num_beams=1, max_new_tokens=200)
        h, r = norm(proc.batch_decode(ids, skip_special_tokens=True)[0]), norm(txt)
        e += ed(list("".join(h)), list("".join(r))); n += len("".join(r)); we += ed(h, r); wn += len(r)
    gen.train()
    return round(e / max(n, 1), 4), round(we / max(wn, 1), 4)

# ---------------------------------------------------------------- training
gen = load_gen(); gen.train()
disc = Discriminators(scale=0.5)
opt_g = torch.optim.AdamW(gen.parameters(), a.lr, betas=(0.8, 0.99))
opt_d = torch.optim.AdamW(disc.parameters(), 2e-4, betas=(0.8, 0.99))
mel = MelLoss()
lengths = np.array([e - s for s, e, _ in utt])
ok = np.flatnonzero(lengths > a.seg + 2)
def batch():
    X, Y = [], []
    for i in rng.choice(ok, a.batch):
        T = min(len(pm[i]), len(utt[i][2]) // HOP)
        st = int(rng.integers(0, T - a.seg))
        X.append(pm[i][st:st + a.seg].float())
        Y.append(utt[i][2][st * HOP:(st + a.seg) * HOP].float() / 32768.0)
    return torch.stack(X), torch.stack(Y)

hist, best = [], None
c0, w0 = val_cer(gen)
log(f"step 0 (released vocoder): val CER {c0} WER {w0}")
best = dict(step=0, cer=c0, wer=w0)
torch.save(gen.state_dict(), os.path.join(a.out, "gen_step0.pt"))
for step in range(1, a.steps + 1):
    x, y = batch()
    y_hat = gen(x)[:, :y.shape[1]]
    out = {}
    r, _ = disc(y); f, _ = disc(y_hat.detach())
    ld = L.gan_discriminator(r, f)
    opt_d.zero_grad(); ld.backward(); opt_d.step(); out["d"] = float(ld)
    lm = mel(y_hat, y)
    lg = MEL_LAMBDA * lm + L.mrstft(y_hat, y)
    if step >= a.adv_from:
        r_s, r_f = disc(y); f_s, f_f = disc(y_hat)
        lg = lg + L.gan_generator(f_s) + 2.0 * L.feature_matching(r_f, f_f)
    opt_g.zero_grad(); lg.backward(); torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0); opt_g.step()
    out.update(g=float(lg), mel=float(lm))
    if not all(np.isfinite(v) for v in out.values()): raise FloatingPointError(f"step {step}: {out}")
    hist.append(out)
    if step % 50 == 0:
        log(f"step {step} mel {np.mean([h['mel'] for h in hist[-50:]]):.3f} d {np.mean([h['d'] for h in hist[-50:]]):.3f}")
    if step % a.eval_every == 0:
        c, w = val_cer(gen)
        log(f"step {step}: val CER {c} WER {w} (best {best['cer']} @ {best['step']})")
        if c < best["cer"]:
            best = dict(step=step, cer=c, wer=w)
            torch.save(gen.state_dict(), os.path.join(a.out, "gen_best.pt"))
json.dump(dict(best=best, released_vocoder=dict(cer=c0, wer=w0), args=vars(a), seconds=round(time.time() - t0)),
          open(os.path.join(a.out, "finetune_report.json"), "w"), indent=1)
log("done", best)
