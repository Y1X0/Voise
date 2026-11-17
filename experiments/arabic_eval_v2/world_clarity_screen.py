#!/usr/bin/env python3
"""WORLD clarity screen, implemented to WORLD_CLARITY_SCREEN_PREREG.md (v1.1, frozen 2026-10-18).

Measures clarity only: relative corpus-WER increase of WORLD pseudo-speaker renders vs the original
Omnilingual Arabic dev audio, for whisper-small and MMS-1B-all. No training, no vocoder, no teacher.

Section references (§) are to the pre-registration. Structure:
  * pure functions (reference preparation, filters, validity guard, population ranking, scoring,
    worst case, bootstrap, decision) -- unit-tested on synthetic data (test_world_clarity_screen.py);
  * run_screen(): the pipeline with every I/O dependency injected (audio fetch, ASR, WORLD class);
  * main(): the real wiring (Hugging Face dataset at the pinned revision, ASR models, Kaggle paths).
    It refuses to run unless VOISE_SCREEN_APPROVED=1 (real-data processing needs owner approval).

v1 metric code is NOT copied: norm() and ed() are loaded verbatim (AST) from teacher_test.py, and
PseudoSpeakerWorld from scripts/neural_anonymization_eval.py; the sha256 of both files is recorded.
"""
import ast
import collections
import hashlib
import io
import json
import os
import re
import shutil
import signal
import sys
import time
import unicodedata

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# ---------------------------------------------------------------- pre-registered constants
PREREG = "experiments/arabic_eval_v2/WORLD_CLARITY_SCREEN_PREREG.md"
REPO = "facebook/omnilingual-asr-corpus"
REVISION = "8648ba8946377697b427ae952076e49fc0e5e44d"                                    # §3
CONFIGS = ("aao_Arab", "abv_Arab", "acm_Arab", "acw_Arab", "acx_Arab", "adf_Arab", "aeb_Arab", "aec_Arab",
           "afb_Arab", "apc_Arab", "apd_Arab", "arq_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "avl_Arab",
           "ayl_Arab", "ayp_Arab")                                                         # §3 (shu_Latn excluded)
EXCLUDED_CONFIGS = ("shu_Latn",)
TAGS = ("<laugh>", "<hesitation>", "<unintelligible>", "<noise>")                         # §4
TAG_RE = re.compile(r"<[^<>]*>")
MAX_DURATION_S = 30.0                                                                      # §5
PRIMARY_RULE = (5, 3, 120.0)                       # §6: >= segments, >= distinct prompts, >= seconds
GUARD_MIN_SPEAKERS = 8                             # §6 HEURISTIC
GUARD_EXPECTED_DEV_SPEAKERS = 11                   # §3
GUARD_MIN_WORDS = 2000                             # §6 HEURISTIC
WORLD_SEED = 4242                                  # §7
POP_SALT = "voise-world-pop-v1"                    # §7
POP_PER_SPEAKER = 3                                # §7 HEURISTIC
POP_MIN_S, POP_MAX_S = 3.0, 30.0                   # §7 HEURISTIC (inclusive)
MIN_VOICED_FRAMES = 10                             # §7 (converter's own skip rule)
REL_THRESHOLD = 0.20                               # §9, data/acceptance_criteria.json spontaneous_wer_rel
BOOT_B, BOOT_SEED = 10_000, 20261018               # §9
ASR_NAMES = ("whisper", "mms")
TRANSFORMERS_PIN = "5.18.0"                        # §5
WORK = "/tmp/voise_screen"                         # §11
HF_CACHE = "/tmp/voise_screen_hf"                  # §11
OUTDIR = "/kaggle/working"                         # §11
# Implementation parameters (not in the pre-registration text; reported in results.json)
WORLD_TIMEOUT_S = 600
ASR_TIMEOUT_S = 600
SR = 16000

FAIL = object()        # sentinel: a processing failure (§9a), never confused with an empty transcript


# ---------------------------------------------------------------- verbatim v1 code loading
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_definitions(path, names, namespace):
    """Exec only the named top-level function/class definitions of `path`, verbatim (no other code
    of that file runs). Returns {name: object}."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    found = {n.name for n in nodes}
    if found != set(names):
        raise RuntimeError(f"{path}: definitions not found: {sorted(set(names) - found)}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return {n: namespace[n] for n in names}


V1_PATH = os.path.join(HERE, "teacher_test.py")
WORLD_PATH = os.path.join(ROOT, "scripts", "neural_anonymization_eval.py")
_v1 = load_definitions(V1_PATH, ("norm", "ed"), {"unicodedata": unicodedata})
norm, ed = _v1["norm"], _v1["ed"]


def load_world_class():
    return load_definitions(WORLD_PATH, ("PseudoSpeakerWorld",), {"np": np, "sf": sf})["PseudoSpeakerWorld"]


# ---------------------------------------------------------------- §4 reference text, §5 duration
def prepare_reference(raw):
    """-> (reference words, None) or (None, exclusion reason). §4 steps 1-3."""
    if not isinstance(raw, str):
        return None, "missing_reference"
    if any(t not in TAGS for t in TAG_RE.findall(raw)):
        return None, "undocumented_tag"
    text = raw
    for t in TAGS:
        text = text.replace(t, " ")
    words = norm(text).split()
    if not words:
        return None, "empty_reference"
    return words, None


def row_key(r):
    return (r["config"], r["speaker_id"], r["prompt_id"], r["segment_id"])


def filter_dev_rows(rows):
    """§4-§5. Exclusions decided from metadata and reference text only. Fixed reason order:
    missing_duration, duration_over_30s, missing_reference, undocumented_tag, empty_reference."""
    kept, excluded = [], collections.Counter()
    excluded_keys = collections.defaultdict(list)
    for r in rows:
        d = r.get("duration")
        reason = None
        if d is None:
            reason = "missing_duration"
        elif float(d) > MAX_DURATION_S:
            reason = "duration_over_30s"
        words = None
        if reason is None:
            words, reason = prepare_reference(r.get("raw_text"))
        if reason:
            excluded[reason] += 1
            excluded_keys[reason].append(row_key(r))
            continue
        kept.append(dict(r, ref_words=words, n_words=len(words),
                         has_unintelligible="<unintelligible>" in r["raw_text"]))
    return kept, dict(excluded), {k: v for k, v in excluded_keys.items()}


def tag_prevalence(rows):
    """§10: per documented tag, rows / occurrences / reference words / speakers affected (all dev rows
    with a text reference; words = §4 reference words of those rows, 0 for rows §4 excludes)."""
    out = {}
    for t in TAGS:
        hit = [r for r in rows if isinstance(r.get("raw_text"), str) and t in r["raw_text"]]
        out[t] = dict(rows=len(hit), occurrences=sum(r["raw_text"].count(t) for r in hit),
                      words=sum(len(prepare_reference(r["raw_text"])[0] or []) for r in hit),
                      speakers=len({(r["config"], r["speaker_id"]) for r in hit}))
    return out


# ---------------------------------------------------------------- §6 validity guard
def speaker_represented(rows):
    segs = {(r["prompt_id"], r["segment_id"]) for r in rows}
    prompts = {r["prompt_id"] for r in rows}
    dur = sum(float(r["duration"]) for r in rows)
    a, b, s = PRIMARY_RULE
    return len(segs) >= a and len(prompts) >= b and dur >= s


def evaluate_guard(rows, dev_speakers):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["config"], r["speaker_id"])].append(r)
    represented = sorted(s for s in dev_speakers if by.get(s) and speaker_represented(by[s]))
    words = int(sum(r["n_words"] for r in rows))
    return dict(dev_speakers=len(dev_speakers), represented_speakers=len(represented), reference_words=words,
                min_speakers=GUARD_MIN_SPEAKERS, min_words=GUARD_MIN_WORDS,
                valid=len(represented) >= GUARD_MIN_SPEAKERS and words >= GUARD_MIN_WORDS,
                represented=[list(s) for s in represented])


# ---------------------------------------------------------------- §7 population
def pop_rank(r):
    s = f"{POP_SALT}|{r['config']}|{r['speaker_id']}|{r['prompt_id']}|{r['segment_id']}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rank_population(train_rows):
    """{speaker key: [rows in hash order]} for rows with POP_MIN_S <= duration <= POP_MAX_S."""
    by = collections.defaultdict(list)
    for r in train_rows:
        d = r.get("duration")
        if d is not None and POP_MIN_S <= float(d) <= POP_MAX_S:
            by[(r["config"], r["speaker_id"])].append(r)
    return {s: sorted(v, key=pop_rank) for s, v in sorted(by.items())}


def select_population(ranked, try_clip):
    """Walk each speaker's ranking until POP_PER_SPEAKER clips are retained. try_clip(row) ->
    (path or None, status) with status in {"retained", "too_few_voiced_frames", "source_failure: ..."}."""
    paths, record, short = [], [], []
    for spk, rows in ranked.items():
        got = 0
        for r in rows:
            if got >= POP_PER_SPEAKER:
                break
            path, status = try_clip(r)
            record.append(dict(key=list(row_key(r)), status=status))
            if status == "retained":
                paths.append(path)
                got += 1
        if got < POP_PER_SPEAKER:
            short.append(dict(speaker=list(spk), retained=got))
    return paths, record, short


def voiced_frames(analyser, path):
    """Voiced-frame count under the converter's own analysis (PseudoSpeakerWorld._analyse)."""
    x, sr = sf.read(path)
    f0 = analyser._analyse(x.astype(np.float64), sr)[1]
    return int((f0 > 0).sum())


def make_analyser(world_cls):
    a = world_cls.__new__(world_cls)
    a.fs, a.fft = SR, 1024
    a.mel = a._mel_matrix()
    return a


# ---------------------------------------------------------------- audio
def decode_to_wav(audio_bytes, path):
    """Decode, mono (channel mean), resample to 16 kHz (torchaudio, as v1), write PCM_16 WAV."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_16")


def validate_wav(path):
    """§9a: non-empty, finite, 16 kHz mono. -> None if valid, else the failing check."""
    try:
        info = sf.info(path)
    except Exception as e:                                   # noqa: BLE001
        return f"unreadable: {type(e).__name__}"
    if info.samplerate != SR:
        return f"sample_rate_{info.samplerate}"
    if info.channels != 1:
        return f"channels_{info.channels}"
    if info.frames <= 0:
        return "empty"
    x, _ = sf.read(path, dtype="float32")
    if not np.all(np.isfinite(x)):
        return "non_finite"
    return None


class Timeout(Exception):
    pass


def call_with_timeout(fn, seconds, *a, **k):
    if seconds is None or not hasattr(signal, "SIGALRM"):
        return fn(*a, **k)

    def handler(signum, frame):
        raise Timeout(f"timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        return fn(*a, **k)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def transcribe(asr_fn, path, timeout=None, faults=None):
    """§9a: one deterministic retry; a remaining fault -> FAIL. Any returned string (also empty) is valid.
    The type name of every faulted attempt (e.g. "Timeout") is appended to `faults` (no messages)."""
    timeout = ASR_TIMEOUT_S if timeout is None else timeout
    x, _ = sf.read(path, dtype="float32")
    for _ in range(2):
        try:
            out = call_with_timeout(asr_fn, timeout, x)
            if not isinstance(out, str):
                raise TypeError("ASR returned a non-string")
            return out
        except Exception as e:                               # noqa: BLE001
            if faults is not None:
                faults.append(type(e).__name__)
            continue
    return FAIL


# ---------------------------------------------------------------- §8-§9a scoring
def row_edits(row, asr, cond):
    """Edits for one condition; None when the row has no usable hypothesis for it (ASR failure).
    A WORLD conversion failure scores the WORLD condition as an empty hypothesis (§9a)."""
    if cond == "world" and not row["world_ok"]:
        return ed(row["ref_words"], [])
    hyp = row["hyp"][asr][cond]
    if hyp is FAIL:
        return None
    return ed(row["ref_words"], norm(hyp).split())


def asr_sets(rows, asr):
    """-> P (primary), F (ASR failed on WORLD only), O (ASR failed on the original)."""
    P, F, O = [], [], []
    for r in rows:
        if r["hyp"][asr]["orig"] is FAIL:
            O.append(r)
        elif r["world_ok"] and r["hyp"][asr]["world"] is FAIL:
            F.append(r)
        else:
            P.append(r)
    return P, F, O


def corpus(rows, edits_o, edits_w):
    words = sum(r["n_words"] for r in rows)
    eo, ew = sum(edits_o), sum(edits_w)
    out = dict(rows=len(rows), words=int(words), edits_orig=int(eo), edits_world=int(ew))
    if words == 0:
        out.update(wer_orig=None, wer_world=None, rel=None, defined=False)
        return out
    wo, ww = eo / words, ew / words
    out.update(wer_orig=wo, wer_world=ww, rel=(ww - wo) / wo if wo > 0 else None, defined=wo > 0)
    return out


def score_asr(rows, asr):
    P, F, O = asr_sets(rows, asr)
    eo_p = [row_edits(r, asr, "orig") for r in P]
    ew_p = [row_edits(r, asr, "world") for r in P]
    primary = corpus(P, eo_p, ew_p)
    wc_rows = P + F
    eo_wc = eo_p + [row_edits(r, asr, "orig") for r in F]
    ew_wc = ew_p + [len(r["ref_words"]) for r in F]          # empty WORLD hypothesis for F rows
    worst = corpus(wc_rows, eo_wc, ew_wc)
    return dict(P=P, F=F, O=O, primary=primary, worst_case=worst, edits_orig=eo_p, edits_world=ew_p)


def outcome_of(rels):
    return "PASS" if all(v <= REL_THRESHOLD for v in rels) else "FAIL"


def decide(pre_guard, per_asr):
    """§9 / §9a. per_asr[a] has 'primary', 'worst_case', 'guard_after_failures'."""
    if not pre_guard["valid"]:
        return "INCONCLUSIVE", "validity guard failed before ASR (§6)"
    for a, s in per_asr.items():
        if not s["guard_after_failures"]["valid"]:
            return "INCONCLUSIVE", f"{a}: guard not met after ASR-failure removal (§9a)"
    for a, s in per_asr.items():
        for name in ("primary", "worst_case"):
            if not s[name]["defined"]:
                return "INCONCLUSIVE", f"{a}: {name} WER_orig = 0 or no valid rows (§9, §9a)"
    p = outcome_of([s["primary"]["rel"] for s in per_asr.values()])
    w = outcome_of([s["worst_case"]["rel"] for s in per_asr.values()])
    if p != w:
        return "INCONCLUSIVE", f"primary {p} vs worst case {w} (§9a)"
    return p, "point estimates (§9)"


# ---------------------------------------------------------------- §9 descriptive bootstrap
def bootstrap(rows, edits_o, edits_w, speakers, B=BOOT_B, seed=BOOT_SEED):
    """teacher_test.py method: multinomial speaker multiplicities, ratio of sums, same draws for
    both conditions. `speakers`: every dev speaker with >= 1 row in the ASR's primary set, so the
    intervals describe the same rows as the point estimate (§9 estimand). Descriptive only."""
    spk = sorted({tuple(s) for s in speakers})
    if not spk:
        return dict(n_speakers=0, note="no speakers")
    idx = {s: i for i, s in enumerate(spk)}
    K = len(spk)
    W, Eo, Ew = np.zeros(K), np.zeros(K), np.zeros(K)
    for r, eo, ew in zip(rows, edits_o, edits_w):
        s = (r["config"], r["speaker_id"])
        if s in idx:
            W[idx[s]] += r["n_words"]; Eo[idx[s]] += eo; Ew[idx[s]] += ew
    rng = np.random.default_rng(seed)
    M = rng.multinomial(K, np.full(K, 1 / K), size=B).astype(float)
    words = M @ W
    wo, ww = (M @ Eo) / words, (M @ Ew) / words
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    k = int(np.sum(wo == 0))
    out = dict(n_speakers=K, B=B, seed=seed, wer_orig_ci95=ci(wo), wer_world_ci95=ci(ww), delta_wer_ci95=ci(ww - wo),
               rel_undefined_resamples=k)
    if k == 0:
        out["rel_ci95"] = ci((ww - wo) / wo)
    else:
        out["rel_ci95"] = None
        out["rel_ci_note"] = f"rel CI not reported: {k} of {B} resamples had WER_orig = 0"
    return out


# ---------------------------------------------------------------- pipeline (I/O injected)
def run_screen(dev_meta, dev_text, dev_speakers, train_meta, fetch_audio, world_cls, asr_factory, workdir,
               log=print, B=BOOT_B):
    """dev_meta/train_meta: metadata rows (config, speaker_id, prompt_id, segment_id, duration).
    dev_text: {row key: raw_text}. fetch_audio(rows) -> {row key: bytes or Exception}.
    asr_factory() -> {"whisper": fn(float32 16 kHz array) -> str, "mms": fn}; called only after the
    guard passes and all WORLD renders exist. Returns the results dict (no audio, no text)."""
    R = dict(prereg=PREREG, status="RUNNING", revision=REVISION, configs=list(CONFIGS),
             excluded_configs=list(EXCLUDED_CONFIGS), implementation_parameters=dict(
                 world_timeout_s=WORLD_TIMEOUT_S, asr_timeout_s=ASR_TIMEOUT_S, asr_retries=1))
    dev_rows = [dict(r, raw_text=dev_text.get(row_key(r))) for r in dev_meta]
    R["tag_prevalence"] = tag_prevalence(dev_rows)
    kept, excl, excl_keys = filter_dev_rows(dev_rows)
    R["dev_filter"] = dict(rows_in=len(dev_rows), rows_kept=len(kept), excluded=excl,
                           excluded_keys={k: [list(x) for x in v] for k, v in excl_keys.items()})
    # source audio (§9a source failures), then the guard, before any ASR
    src_dir = os.path.join(workdir, "src")
    blobs = fetch_audio(kept)
    rows, src_fail = [], []
    for r in kept:
        k = row_key(r)
        path = os.path.join(src_dir, hashlib.sha256(repr(k).encode()).hexdigest()[:24] + ".wav")
        b = blobs.get(k)
        err = None
        if b is None or isinstance(b, Exception):
            err = f"fetch: {type(b).__name__ if b is not None else 'missing'}"
        else:
            try:
                decode_to_wav(b, path)
                err = validate_wav(path)
            except Exception as e:                           # noqa: BLE001
                err = f"decode: {type(e).__name__}"
        if err:
            src_fail.append(dict(key=list(k), check=err))
            continue
        rows.append(dict(r, src=path, src_sha256=sha256_file(path)))
    R["source_failures"] = src_fail
    guard = evaluate_guard(rows, dev_speakers)
    R["validity_guard_before_asr"] = guard
    log(f"guard before ASR: valid={guard['valid']} represented={guard['represented_speakers']} words={guard['reference_words']}")
    if not guard["valid"]:
        R.update(status="DONE", result="INCONCLUSIVE", reason="validity guard failed before ASR (§6); no ASR run")
        return R
    # population (§7)
    pop_dir = os.path.join(workdir, "pop")
    analyser = make_analyser(world_cls)
    ranked = rank_population(train_meta)

    def try_clip(r):
        k = row_key(r)
        b = fetch_audio([r]).get(k)
        if b is None or isinstance(b, Exception):
            return None, "source_failure: fetch"
        path = os.path.join(pop_dir, hashlib.sha256(repr(k).encode()).hexdigest()[:24] + ".wav")
        try:
            decode_to_wav(b, path)
            bad = validate_wav(path)
            if bad:
                return None, f"source_failure: {bad}"
            if voiced_frames(analyser, path) < MIN_VOICED_FRAMES:
                return None, "too_few_voiced_frames"
        except Exception as e:                               # noqa: BLE001
            return None, f"source_failure: {type(e).__name__}"
        return path, "retained"
    pop_paths, pop_record, pop_short = select_population(ranked, try_clip)
    R["population"] = dict(train_speakers=len(ranked), retained_clips=len(pop_paths), record=pop_record,
                           speakers_short=pop_short)
    if not pop_paths:
        R.update(status="STOPPED", result="INCONCLUSIVE", reason="hard stop: no population clip retained (§7)")
        return R
    world = world_cls(pop_paths, cmvn=True)
    # WORLD renders (§7, §9a)
    w_dir = os.path.join(workdir, "world")
    os.makedirs(w_dir, exist_ok=True)
    for r in rows:
        out = os.path.join(w_dir, os.path.basename(r["src"]))
        try:
            call_with_timeout(world.run, WORLD_TIMEOUT_S, r["src"], out, WORLD_SEED, "neutral")
            bad = validate_wav(out)
        except Exception as e:                               # noqa: BLE001
            bad = f"{type(e).__name__}"
        r["world_ok"] = bad is None
        r["world"] = out if bad is None else None
        r["world_check"] = bad
        r["world_sha256"] = sha256_file(out) if bad is None else None
    # ASR (§8)
    asr = asr_factory()
    faults = {a: [] for a in ASR_NAMES}
    for r in rows:
        r["hyp"] = {}
        for a in ASR_NAMES:
            r["hyp"][a] = {}
            for cond, path in (("orig", r["src"]), ("world", r["world"])):
                if cond == "world" and not r["world_ok"]:
                    r["hyp"][a][cond] = None
                    continue
                f = []
                r["hyp"][a][cond] = transcribe(asr[a], path, faults=f)
                if f:
                    faults[a].append(dict(key=list(row_key(r)), condition=cond, attempt_faults=f,
                                          failed=r["hyp"][a][cond] is FAIL))
    per = {}
    for a in ASR_NAMES:
        s = score_asr(rows, a)
        g = evaluate_guard(s["P"], dev_speakers)
        per[a] = dict(primary=s["primary"], worst_case=s["worst_case"], guard_after_failures=g,
                      failures=dict(asr_fail_orig=[list(row_key(r)) for r in s["O"]],
                                    asr_fail_world=[list(row_key(r)) for r in s["F"]],
                                    asr_faulted_attempts=faults[a],
                                    asr_timeouts=sum(x["attempt_faults"].count("Timeout") for x in faults[a])),
                      bootstrap=bootstrap(s["P"], s["edits_orig"], s["edits_world"],
                                          sorted({(r["config"], r["speaker_id"]) for r in s["P"]}), B=B))
        sens = [i for i, r in enumerate(s["P"]) if not r["has_unintelligible"]]
        per[a]["sensitivity_without_unintelligible"] = corpus([s["P"][i] for i in sens], [s["edits_orig"][i] for i in sens],
                                                              [s["edits_world"][i] for i in sens])
        per[a]["per_row"] = [dict(key=list(row_key(r)), words=r["n_words"], edits_orig=eo, edits_world=ew)
                             for r, eo, ew in zip(s["P"], s["edits_orig"], s["edits_world"])]
    R["world_conversion_failures"] = [dict(key=list(row_key(r)), status="world_conversion_failed", check=r["world_check"])
                                      for r in rows if not r["world_ok"]]
    R["world_timeouts"] = sum(r["world_check"] == "Timeout" for r in rows)
    R["hashes"] = [dict(key=list(row_key(r)), src_sha256=r["src_sha256"], world_sha256=r["world_sha256"]) for r in rows]
    R["per_asr"] = per
    R["result"], R["reason"] = decide(guard, per)
    R["status"] = "DONE"
    return R


# ---------------------------------------------------------------- real wiring (Kaggle; owner approval required)
OPENED_SPLITS = ("dev", "train")                   # §3 v1.1: the official test split is never opened
META_COLS = ["speaker_id", "prompt_id", "segment_id", "duration"]
TEST_SPLIT_NOTE = ("official test split not opened (§3 v1.1); dev/test and train/test speaker separation "
                   "NOT re-verified by this run (census y1x00r/omni-arabic-census reported 0 speakers in "
                   "more than one split)")


def split_of(path):
    return path.split("/")[-1].split("-")[0]


def read_parquet_columns(fs, path, cols):
    import pyarrow.parquet as pq
    return pq.ParquetFile(fs.open(path, block_size=1 << 16)).read(columns=cols).to_pydict()


def read_metadata(fs, base, read_table=read_parquet_columns):
    """§3 v1.1: list each config's parquet files by name and open ONLY dev/train files; files of any
    other split (the official test split) are skipped by name before anything opens them.
    -> ({(config, split): [paths]}, {config: [metadata rows]}, sorted skipped split names)."""
    files, meta, skipped = {}, collections.defaultdict(list), set()
    for cfg in CONFIGS:
        for f in sorted(p for p in fs.ls(f"{base}/{cfg}", detail=False) if p.endswith(".parquet")):
            split = split_of(f)
            if split not in OPENED_SPLITS:
                skipped.add(split)
                continue
            files.setdefault((cfg, split), []).append(f)
            t = read_table(fs, f, META_COLS)
            for i in range(len(t["speaker_id"])):
                meta[cfg].append(dict(config=cfg, split=split, **{c: t[c][i] for c in META_COLS}))
    return files, meta, sorted(skipped)


def census_expected(census):
    """Census counts restricted to the opened splits (§3 v1.1)."""
    exp = {}
    for cfg in CONFIGS:
        pc = census["per_config"][cfg]
        exp[cfg] = dict(rows={k: v for k, v in pc["rows_by_split"].items() if k in OPENED_SPLITS},
                        speakers={k: v for k, v in pc["speakers_by_split_membership"].items() if k in OPENED_SPLITS})
    return exp


def check_integrity(meta_by_config, census):
    """§3 v1.1: per-config dev and train row and speaker counts equal the census; no speaker in both dev
    and train. Only dev/train metadata is used; separation from the test split is not re-verified."""
    problems = []
    if census.get("revision") != REVISION:
        problems.append(f"census revision {census.get('revision')} != {REVISION}")
    for cfg, exp in census_expected(census).items():
        rows = meta_by_config.get(cfg, [])
        got_rows = dict(collections.Counter(r["split"] for r in rows))
        spl = collections.defaultdict(set)
        for r in rows:
            spl[r["speaker_id"]].add(r["split"])
        multi = [s for s, v in spl.items() if len(v) > 1]
        got_spk = dict(collections.Counter(next(iter(v)) for v in spl.values() if len(v) == 1))
        if got_rows != exp["rows"]:
            problems.append(f"{cfg}: rows {got_rows} != {exp['rows']}")
        if got_spk != exp["speakers"]:
            problems.append(f"{cfg}: speakers {got_spk} != {exp['speakers']}")
        if multi:
            problems.append(f"{cfg}: {len(multi)} speaker(s) in more than one split")
    return problems


def cleanup(paths=(WORK, HF_CACHE), outdir=OUTDIR, log=print):
    """§11: delete the run-local paths recursively; print what remains. Our own evidence only."""
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)
    state = {p: os.path.exists(p) for p in paths}
    listing = sorted(os.listdir(outdir)) if os.path.isdir(outdir) else None
    log(f"cleanup: still_exists={state} output_listing={listing}")
    return state, listing


def main():
    if os.environ.get("VOISE_SCREEN_APPROVED") != "1":
        sys.exit("refusing to run on real data: VOISE_SCREEN_APPROVED=1 is required (owner approval, Stage 3)")
    os.environ["HF_HOME"] = HF_CACHE                    # §11: before any Hugging Face import
    os.makedirs(WORK, exist_ok=True)
    import pyarrow.parquet as pq
    import torch
    import transformers
    from huggingface_hub import HfFileSystem
    if transformers.__version__ != TRANSFORMERS_PIN:
        sys.exit(f"transformers {transformers.__version__} != pinned {TRANSFORMERS_PIN} (§5)")
    crit = json.load(open(os.path.join(ROOT, "data", "acceptance_criteria.json")))
    assert crit["criteria"]["spontaneous_wer_rel"]["threshold"] == REL_THRESHOLD
    census = json.load(open(os.path.join(HERE, "census_results.json")))
    fs = HfFileSystem()
    base = f"datasets/{REPO}@{REVISION}/data"
    meta_cols = META_COLS
    files, meta, skipped_splits = read_metadata(fs, base)
    problems = check_integrity(meta, census)
    if problems:
        sys.exit("integrity check failed (§3): " + "; ".join(problems))
    dev_meta = [r for c in CONFIGS for r in meta[c] if r["split"] == "dev"]
    train_meta = [r for c in CONFIGS for r in meta[c] if r["split"] == "train"]
    dev_speakers = sorted({(r["config"], r["speaker_id"]) for r in dev_meta})
    assert len(dev_speakers) == GUARD_EXPECTED_DEV_SPEAKERS, len(dev_speakers)

    def read_column(cfg, split, keys, column):
        if split not in OPENED_SPLITS:
            raise RuntimeError(f"refusing to open split {split!r} (§3 v1.1)")
        out = {}
        for f in files[(cfg, split)]:
            pf = pq.ParquetFile(fs.open(f, block_size=1 << 20))
            for g in range(pf.num_row_groups):
                t = pf.read_row_group(g, columns=meta_cols[:3]).to_pydict()
                hit = [i for i in range(len(t["speaker_id"]))
                       if (cfg, t["speaker_id"][i], t["prompt_id"][i], t["segment_id"][i]) in keys]
                if not hit:
                    continue
                col = pf.read_row_group(g, columns=[column]).column(column).to_pylist()
                for i in hit:
                    out[(cfg, t["speaker_id"][i], t["prompt_id"][i], t["segment_id"][i])] = col[i]
        return out
    dev_text = {}
    for cfg in sorted({r["config"] for r in dev_meta}):
        dev_text.update(read_column(cfg, "dev", {row_key(r) for r in dev_meta if r["config"] == cfg}, "raw_text"))

    def fetch_audio(rows):
        out, want = {}, collections.defaultdict(set)
        for r in rows:
            want[(r["config"], r["split"])].add(row_key(r))
        for (cfg, split), keys in want.items():
            try:
                got = read_column(cfg, split, keys, "audio")
            except Exception as e:                           # noqa: BLE001
                got = {k: e for k in keys}
            for k in keys:
                v = got.get(k)
                out[k] = (v.get("bytes") if isinstance(v, dict) else v) if not isinstance(v, Exception) else v
        return out

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    revs = {}

    def asr_factory():
        from transformers import AutoProcessor, Wav2Vec2ForCTC, WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained("openai/whisper-small")
        wh = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()
        mp = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="ara")
        mm = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="ara", ignore_mismatched_sizes=True).to(dev).eval()
        revs.update(whisper=getattr(wh.config, "_commit_hash", None), mms=getattr(mm.config, "_commit_hash", None))

        def whisper(a):                                      # teacher_test.py:157-160
            with torch.no_grad():
                f = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
                return proc.batch_decode(wh.generate(f, language="ar", task="transcribe"), skip_special_tokens=True)[0]

        def mms(a):                                          # teacher_test.py:161-162
            with torch.no_grad():
                x = mp(a, sampling_rate=16000, return_tensors="pt").to(dev)
                return mp.decode(mm(**x).logits[0].argmax(-1))
        return dict(whisper=whisper, mms=mms)

    t0 = time.time()
    R = run_screen(dev_meta, dev_text, dev_speakers, train_meta, fetch_audio, load_world_class(), asr_factory, WORK)
    import platform
    import pyworld
    R["integrity_check"] = dict(splits_opened=list(OPENED_SPLITS), splits_skipped_unopened=skipped_splits,
                                test_split=TEST_SPLIT_NOTE)
    R["environment"] = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                            pyworld=getattr(pyworld, "__version__", None), numpy=np.__version__, soundfile=sf.__version__,
                            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, asr_revisions=revs,
                            seconds=round(time.time() - t0, 1))
    R["code_sha256"] = {"world_clarity_screen.py": sha256_file(os.path.abspath(__file__)),
                        "teacher_test.py": sha256_file(V1_PATH), "neural_anonymization_eval.py": sha256_file(WORLD_PATH),
                        "WORLD_CLARITY_SCREEN_PREREG.md": sha256_file(os.path.join(ROOT, PREREG))}
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=str)
    print(f"results.json sha256 {sha256_file(out)}; result {R.get('result')}: {R.get('reason')}", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
