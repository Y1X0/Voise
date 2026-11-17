"""Synthetic tests for world_clarity_screen.py (WORLD_CLARITY_SCREEN_PREREG.md DRAFT v0.3).

No dataset rows, audio or transcripts are used: every row, reference and waveform here is generated.
No network, no ASR model (fake ASR functions), no training. The end-to-end test runs the real
PseudoSpeakerWorld converter (pyworld) on synthetic tones.
"""
import ast
import collections
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world_clarity_screen as W  # noqa: E402

SR = 16000
VOCAB = ["كتاب", "بيت", "شمس", "قمر", "باب", "نهر", "جبل", "ورد", "سماء", "بحر"]


def tone(seconds, f0=140.0, sr=SR, channels=1, extra=0, seed=0):
    """Voiced harmonic tone with vibrato and a little noise (synthetic, not speech)."""
    n = int(seconds * sr) + extra
    t = np.arange(n) / sr
    f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, 8)) * 0.2
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t) ** 2)
    x = x + 0.003 * np.random.default_rng(seed).standard_normal(n)
    x = x.astype(np.float32)
    return np.stack([x] * channels, 1) if channels > 1 else x


def wav_bytes(x, sr=SR, subtype="PCM_16"):
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype=subtype)
    return b.getvalue()


def row(cfg="arz_Arab", spk="s1", prompt="p1", seg="g1", dur=1.0, text=None, split="dev"):
    r = dict(config=cfg, speaker_id=spk, prompt_id=prompt, segment_id=seg, duration=dur, split=split)
    if text is not None:
        r["raw_text"] = text
    return r


def scored_row(spk, n_words, orig, world, world_ok=True, cfg="arz_Arab", seg="g"):
    """A row as score_asr sees it; orig/world: {asr: hypothesis string or W.FAIL}."""
    ref = VOCAB[:n_words]
    return dict(config=cfg, speaker_id=spk, prompt_id="p", segment_id=seg, ref_words=ref, n_words=n_words,
                world_ok=world_ok, has_unintelligible=False,
                hyp={a: dict(orig=orig[a], world=world[a] if world_ok else None) for a in W.ASR_NAMES})


# ------------------------------------------------------------------ v1 code is loaded, not copied
class TestVerbatimV1(unittest.TestCase):
    def test_norm_ed_come_from_teacher_test(self):
        tree = ast.parse(open(W.V1_PATH, encoding="utf-8").read())
        lines = {n.name: n.lineno for n in tree.body if isinstance(n, ast.FunctionDef)}
        for fn in (W.norm, W.ed):
            self.assertEqual(fn.__code__.co_filename, W.V1_PATH)
            self.assertEqual(fn.__code__.co_firstlineno, lines[fn.__name__])
        src = open(W.__file__, encoding="utf-8").read()
        self.assertNotIn("def norm(", src)
        self.assertNotIn("def ed(", src)

    def test_world_class_comes_from_converter_file(self):
        cls = W.load_world_class()
        self.assertEqual(cls.run.__code__.co_filename, W.WORLD_PATH)
        self.assertNotIn("class PseudoSpeakerWorld", open(W.__file__, encoding="utf-8").read())

    def test_norm_ed_behaviour(self):
        self.assertEqual(W.norm("أحمد إلى آخر ٱلبيت مدرسة"), W.norm("احمد الى اخر البيت مدرسه"))
        self.assertEqual(W.ed(["a", "b", "c"], ["a", "c"]), 1)
        self.assertEqual(W.ed(["a", "b"], []), 2)
        self.assertEqual(W.ed([], ["x"]), 1)

    def test_loader_rejects_missing_names(self):
        with self.assertRaises(RuntimeError):
            W.load_definitions(W.V1_PATH, ("norm", "no_such_function"), {})

    def test_threshold_matches_frozen_criteria(self):
        crit = json.load(open(os.path.join(W.ROOT, "data", "acceptance_criteria.json")))
        self.assertEqual(crit["criteria"]["spontaneous_wer_rel"]["threshold"], W.REL_THRESHOLD)


# ------------------------------------------------------------------ §4-§5 reference and filters
class TestReference(unittest.TestCase):
    def test_tags_removed_and_normalised(self):
        w, why = W.prepare_reference("كتاب <laugh> بيت <noise><hesitation> شمس <unintelligible>")
        self.assertIsNone(why)
        self.assertEqual(w, ["كتاب", "بيت", "شمس"])

    def test_exclusion_reasons(self):
        self.assertEqual(W.prepare_reference(None), (None, "missing_reference"))
        self.assertEqual(W.prepare_reference("كتاب <music> بيت"), (None, "undocumented_tag"))
        self.assertEqual(W.prepare_reference("<laugh> <noise>"), (None, "empty_reference"))
        self.assertEqual(W.prepare_reference("123 ، !"), (None, "empty_reference"))

    def test_filter_reason_order_and_boundary(self):
        rows = [row(seg="a", dur=None, text=None),                 # missing_duration wins over missing_reference
                row(seg="b", dur=31.0, text="كتاب <music>"),       # duration_over_30s wins over undocumented_tag
                row(seg="c", dur=30.0, text="كتاب بيت"),           # 30.0 s is kept (§5: > 30 s excluded)
                row(seg="d", dur=30.0001, text="كتاب"),
                row(seg="e", dur=2.0),
                row(seg="f", dur=2.0, text="باب <unintelligible>")]
        kept, excl, keys = W.filter_dev_rows(rows)
        self.assertEqual(excl, {"missing_duration": 1, "duration_over_30s": 2, "missing_reference": 1})
        self.assertEqual([r["segment_id"] for r in kept], ["c", "f"])
        self.assertEqual(kept[0]["n_words"], 2)
        self.assertEqual([kept[0]["has_unintelligible"], kept[1]["has_unintelligible"]], [False, True])
        self.assertEqual(keys["missing_duration"], [("arz_Arab", "s1", "p1", "a")])

    def test_tag_prevalence(self):
        rows = [row(seg="a", text="كتاب <laugh> <laugh>"), row(spk="s2", seg="b", text="بيت شمس <laugh>"),
                row(seg="c", text="قمر <music> <laugh>"), row(seg="d")]
        p = W.tag_prevalence(rows)
        self.assertEqual(p["<laugh>"], dict(rows=3, occurrences=4, words=3, speakers=2))
        self.assertEqual(p["<noise>"], dict(rows=0, occurrences=0, words=0, speakers=0))


# ------------------------------------------------------------------ §6 guard
def speaker_rows(spk, n_seg, n_prompts, seconds_each, words_each=1, cfg="arz_Arab"):
    return [dict(row(cfg=cfg, spk=spk, prompt=f"p{i % n_prompts}", seg=f"g{i}", dur=seconds_each),
                 n_words=words_each) for i in range(n_seg)]


class TestGuard(unittest.TestCase):
    def test_primary_rule_boundaries(self):
        self.assertTrue(W.speaker_represented(speaker_rows("s", 5, 3, 24.0)))
        self.assertFalse(W.speaker_represented(speaker_rows("s", 4, 3, 30.0)))
        self.assertFalse(W.speaker_represented(speaker_rows("s", 5, 2, 30.0)))
        self.assertFalse(W.speaker_represented(speaker_rows("s", 5, 3, 23.99)))

    def test_duplicate_segments_do_not_count_twice(self):
        rs = speaker_rows("s", 5, 3, 30.0)
        rs[4] = dict(rs[0])
        self.assertFalse(W.speaker_represented(rs))

    def test_guard(self):
        spks = [("arz_Arab", f"s{i}") for i in range(11)]
        rows = [r for i in range(8) for r in speaker_rows(f"s{i}", 5, 3, 24.0, words_each=50)]
        g = W.evaluate_guard(rows, spks)
        self.assertEqual((g["represented_speakers"], g["reference_words"], g["valid"]), (8, 2000, True))
        g = W.evaluate_guard(rows[:-1], spks)                               # 7 speakers
        self.assertFalse(g["valid"])
        rows2 = [dict(r, n_words=49) for r in rows]                         # 1960 words
        self.assertFalse(W.evaluate_guard(rows2, spks)["valid"])
        rows3 = rows + speaker_rows("x", 5, 3, 24.0, words_each=50)         # not a dev speaker
        self.assertEqual(W.evaluate_guard(rows3, spks)["represented_speakers"], 8)


# ------------------------------------------------------------------ §7 population
class TestPopulation(unittest.TestCase):
    def test_rank_is_hash_order_and_input_order_free(self):
        rows = [row(spk=f"t{i % 3}", prompt=f"p{i}", seg=f"g{i}", dur=5.0, split="train") for i in range(12)]
        a = W.rank_population(rows)
        b = W.rank_population(list(reversed(rows)))
        self.assertEqual({k: [W.row_key(r) for r in v] for k, v in a.items()},
                         {k: [W.row_key(r) for r in v] for k, v in b.items()})
        for v in a.values():
            hs = [hashlib.sha256(f"voise-world-pop-v1|{r['config']}|{r['speaker_id']}|{r['prompt_id']}|"
                                 f"{r['segment_id']}".encode()).hexdigest() for r in v]
            self.assertEqual(hs, sorted(hs))

    def test_duration_window_inclusive(self):
        rows = [row(spk="t", seg=s, dur=d, split="train")
                for s, d in [("a", 2.99), ("b", 3.0), ("c", 30.0), ("d", 30.01), ("e", None)]]
        got = [r["segment_id"] for r in W.rank_population(rows)[("arz_Arab", "t")]]
        self.assertEqual(sorted(got), ["b", "c"])

    def test_walk_down_and_shortfall(self):
        ranked = {("c", "t1"): [row(spk="t1", seg=f"g{i}") for i in range(6)],
                  ("c", "t2"): [row(spk="t2", seg=f"g{i}") for i in range(3)]}
        bad = {("arz_Arab", "t1", "p1", "g0"): "too_few_voiced_frames",
               ("arz_Arab", "t2", "p1", "g1"): "source_failure: fetch"}
        calls = []

        def try_clip(r):
            calls.append(W.row_key(r))
            s = bad.get(W.row_key(r), "retained")
            return (str(W.row_key(r)) if s == "retained" else None), s
        paths, rec, short = W.select_population(ranked, try_clip)
        self.assertEqual(len(paths), 5)
        self.assertEqual([c[3] for c in calls if c[1] == "t1"], ["g0", "g1", "g2", "g3"])   # stops at 3 retained
        self.assertEqual(short, [dict(speaker=["c", "t2"], retained=2)])
        self.assertEqual([r["status"] for r in rec][:2], ["too_few_voiced_frames", "retained"])


# ------------------------------------------------------------------ audio, validation, retry
class TestAudio(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        W.cleanup([self.d], outdir=self.d, log=lambda *_: None)

    def p(self, name):
        return os.path.join(self.d, name)

    def test_decode_mono_resample(self):
        W.decode_to_wav(wav_bytes(tone(1.0, sr=22050, channels=2), sr=22050), self.p("x/a.wav"))
        info = sf.info(self.p("x/a.wav"))
        self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))
        self.assertLess(abs(info.frames - 16000), 2)
        self.assertIsNone(W.validate_wav(self.p("x/a.wav")))

    def test_validate_failures(self):
        sf.write(self.p("sr.wav"), tone(0.2), 22050)
        sf.write(self.p("st.wav"), tone(0.2, channels=2), SR)
        sf.write(self.p("empty.wav"), np.zeros(0, np.float32), SR)
        x = tone(0.2)
        x[5] = np.nan
        sf.write(self.p("nan.wav"), x, SR, subtype="FLOAT")
        open(self.p("junk.wav"), "wb").write(b"not audio")
        self.assertEqual(W.validate_wav(self.p("sr.wav")), "sample_rate_22050")
        self.assertEqual(W.validate_wav(self.p("st.wav")), "channels_2")
        self.assertEqual(W.validate_wav(self.p("empty.wav")), "empty")
        self.assertEqual(W.validate_wav(self.p("nan.wav")), "non_finite")
        self.assertTrue(W.validate_wav(self.p("junk.wav")).startswith("unreadable"))
        self.assertTrue(W.validate_wav(self.p("missing.wav")).startswith("unreadable"))

    def test_transcribe_retry_fail_and_empty(self):
        sf.write(self.p("a.wav"), tone(0.2), SR)
        n = collections.Counter()

        def flaky(x):
            n["f"] += 1
            if n["f"] == 1:
                raise RuntimeError("transient")
            return "نص"

        def broken(x):
            n["b"] += 1
            raise RuntimeError("always")
        self.assertEqual(W.transcribe(flaky, self.p("a.wav")), "نص")
        self.assertIs(W.transcribe(broken, self.p("a.wav")), W.FAIL)
        self.assertEqual(n["b"], 2)                                          # one retry, no more
        self.assertEqual(W.transcribe(lambda x: "", self.p("a.wav")), "")   # empty output is valid
        self.assertIs(W.transcribe(lambda x: None, self.p("a.wav")), W.FAIL)

    def test_transcribe_timeout(self):
        sf.write(self.p("a.wav"), tone(0.2), SR)
        t0, f = time.time(), []
        self.assertIs(W.transcribe(lambda x: time.sleep(5) or "x", self.p("a.wav"), timeout=1, faults=f), W.FAIL)
        self.assertLess(time.time() - t0, 4)
        self.assertEqual(f, ["Timeout", "Timeout"])                            # the retry gets its own limit

    def test_cleanup(self):
        a, b, out = self.p("work"), self.p("hf"), self.p("out")
        for d in (a, b, out):
            os.makedirs(os.path.join(d, "sub"))
        open(os.path.join(a, "sub", "f.wav"), "w").write("x")
        open(os.path.join(out, "results.json"), "w").write("{}")
        state, listing = W.cleanup([a, b], outdir=out, log=lambda *_: None)
        self.assertEqual(state, {a: False, b: False})
        self.assertEqual(listing, ["results.json", "sub"])


# ------------------------------------------------------------------ §8-§9a scoring and decision
def S(**k):
    return {a: k.get(a, k.get("both")) for a in W.ASR_NAMES}


class TestScoring(unittest.TestCase):
    def test_sets_and_worst_case(self):
        full = " ".join(VOCAB[:4])
        rows = [scored_row("a", 4, S(both=" ".join(VOCAB[1:4])), S(both=" ".join(VOCAB[2:4]))),   # P: 1 / 2
                scored_row("a", 4, S(both=full), S(both=""), world_ok=False),                     # WORLD conv fail: P, 0 / 4
                scored_row("b", 4, S(both=full), S(whisper=W.FAIL, mms=full)),                    # whisper F
                scored_row("b", 4, S(whisper=W.FAIL, mms=full), S(both=full)),                    # whisper O
                scored_row("b", 4, S(both=full), S(both=""))]                                     # empty hyp: valid, 4 edits
        s = W.score_asr(rows, "whisper")
        self.assertEqual((len(s["P"]), len(s["F"]), len(s["O"])), (3, 1, 1))
        self.assertEqual((s["primary"]["edits_orig"], s["primary"]["edits_world"], s["primary"]["words"]), (1, 10, 12))
        self.assertAlmostEqual(s["primary"]["rel"], (10 / 12 - 1 / 12) / (1 / 12))
        wc = s["worst_case"]
        self.assertEqual((wc["edits_orig"], wc["edits_world"], wc["words"], wc["rows"]), (1, 14, 16, 4))
        m = W.score_asr(rows, "mms")
        self.assertEqual((len(m["P"]), len(m["F"]), len(m["O"])), (5, 0, 0))
        self.assertEqual(m["primary"], m["worst_case"])                     # F empty -> worst case = primary

    def test_world_failure_never_improves(self):
        full = " ".join(VOCAB[:3])
        good = W.score_asr([scored_row("a", 3, S(both=full), S(both=full))], "mms")["primary"]
        fail = W.score_asr([scored_row("a", 3, S(both=full), S(both=full), world_ok=False)], "mms")["primary"]
        self.assertEqual(good["edits_world"], 0)
        self.assertEqual(fail["edits_world"], 3)

    def test_wer_orig_zero_undefined(self):
        c = W.corpus([dict(n_words=5)], [0], [2])
        self.assertFalse(c["defined"])
        self.assertIsNone(c["rel"])
        self.assertFalse(W.corpus([], [], [])["defined"])

    def _per(self, p, w, g=True):
        ok = dict(valid=g)
        return {a: dict(primary=dict(defined=True, rel=p[i]), worst_case=dict(defined=True, rel=w[i]),
                        guard_after_failures=ok) for i, a in enumerate(W.ASR_NAMES)}

    def test_decide(self):
        g = dict(valid=True)
        self.assertEqual(W.decide(g, self._per([0.1, 0.2], [0.15, 0.2]))[0], "PASS")      # 0.20 passes (<=)
        self.assertEqual(W.decide(g, self._per([0.1, 0.21], [0.1, 0.3]))[0], "FAIL")
        self.assertEqual(W.decide(g, self._per([0.1, 0.1], [0.1, 0.25]))[0], "INCONCLUSIVE")  # disagreement
        self.assertEqual(W.decide(dict(valid=False), self._per([0.1, 0.1], [0.1, 0.1]))[0], "INCONCLUSIVE")
        self.assertEqual(W.decide(g, self._per([0.1, 0.1], [0.1, 0.1], g=False))[0], "INCONCLUSIVE")
        per = self._per([0.1, 0.1], [0.1, 0.1])
        per["mms"]["worst_case"] = dict(defined=False, rel=None)
        self.assertEqual(W.decide(g, per)[0], "INCONCLUSIVE")


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(config="c", speaker_id=s, n_words=n) for s, n in
                     [("a", 10), ("a", 5), ("b", 8), ("c", 12), ("z", 100)]]
        self.eo = [1, 0, 2, 3, 50]
        self.ew = [3, 2, 2, 6, 90]
        self.rep = [["c", "a"], ["c", "b"], ["c", "c"]]          # z is not in the speaker list: never resampled

    def test_matches_manual_same_draws(self):
        out = W.bootstrap(self.rows, self.eo, self.ew, self.rep, B=2000, seed=11)
        Wd, Eo, Ew = np.array([15, 8, 12.]), np.array([1, 2, 3.]), np.array([5, 2, 6.])
        M = np.random.default_rng(11).multinomial(3, [1 / 3] * 3, size=2000).astype(float)
        wo, ww = M @ Eo / (M @ Wd), M @ Ew / (M @ Wd)
        pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        self.assertEqual(out["wer_orig_ci95"], pc(wo))
        self.assertEqual(out["wer_world_ci95"], pc(ww))
        self.assertEqual(out["delta_wer_ci95"], pc(ww - wo))                 # same draws for both conditions
        self.assertEqual(out["rel_ci95"], pc((ww - wo) / wo))
        self.assertEqual((out["n_speakers"], out["rel_undefined_resamples"]), (3, 0))
        self.assertEqual(out, W.bootstrap(self.rows, self.eo, self.ew, self.rep, B=2000, seed=11))

    def test_k_positive_reports_no_rel_ci(self):
        eo = [0, 0, 2, 3, 50]                                               # speaker a has WER_orig = 0
        out = W.bootstrap(self.rows, eo, self.ew, self.rep, B=3000, seed=5)
        k = out["rel_undefined_resamples"]
        self.assertGreater(k, 0)
        self.assertIsNone(out["rel_ci95"])
        self.assertEqual(out["rel_ci_note"], f"rel CI not reported: {k} of 3000 resamples had WER_orig = 0")
        self.assertEqual(len(out["delta_wer_ci95"]), 2)                     # always defined
        self.assertTrue(all(np.isfinite(out["wer_orig_ci95"] + out["wer_world_ci95"] + out["delta_wer_ci95"])))

    def test_defaults_are_preregistered(self):
        out = W.bootstrap(self.rows, self.eo, self.ew, self.rep)
        self.assertEqual((out["B"], out["seed"]), (10000, 20261018))


# ------------------------------------------------------------------ §3 integrity
class TestIntegrity(unittest.TestCase):
    def setUp(self):
        # census counts include the test split; the run never sees test metadata (§3 v1.1)
        self.census = dict(revision=W.REVISION, per_config={
            c: dict(rows_by_split={"dev": 2, "test": 4, "train": 3},
                    speakers_by_split_membership={"dev": 1, "test": 1, "train": 1}) for c in W.CONFIGS})
        self.meta = {c: [dict(split="dev", speaker_id="d")] * 2 + [dict(split="train", speaker_id="t")] * 3
                     for c in W.CONFIGS}

    def test_ok_from_dev_and_train_only(self):
        self.assertEqual(W.check_integrity(self.meta, self.census), [])

    def test_mismatches(self):
        m = dict(self.meta)
        m["arz_Arab"] = m["arz_Arab"][:-1]
        m["apc_Arab"] = m["apc_Arab"] + [dict(split="dev", speaker_id="t")]   # speaker in dev and train
        probs = W.check_integrity(m, dict(self.census, revision="other"))
        self.assertTrue(any(p.startswith("census revision") for p in probs))
        self.assertTrue(any(p.startswith("arz_Arab: rows") for p in probs))
        self.assertTrue(any("more than one split" in p and p.startswith("apc_Arab") for p in probs))
        self.assertNotIn("shu_Latn", W.CONFIGS)

    def test_test_split_files_are_never_opened(self):
        base = "datasets/x@rev/data"
        listing = {f"{base}/{c}": [f"{base}/{c}/{s}-00000-of-00001.parquet" for s in ("dev", "test", "train")]
                   + [f"{base}/{c}/README.md"] for c in W.CONFIGS}

        class FS:
            opened = []

            def ls(self, path, detail=False):
                return listing[path]

            def open(self, path, **k):
                self.opened.append(path)
                if W.split_of(path) == "test":
                    raise AssertionError(f"test split opened: {path}")
                return io.BytesIO(b"")
        read = []

        def read_table(fs, path, cols):                 # the injected reader; it must open via fs
            fs.open(path, block_size=1 << 16)
            read.append(path)
            self.assertEqual(cols, ["speaker_id", "prompt_id", "segment_id", "duration"])
            spk = "d" if W.split_of(path) == "dev" else "t"
            return dict(speaker_id=[spk], prompt_id=["p"], segment_id=["g"], duration=[4.0])
        fs = FS()
        files, meta, skipped = W.read_metadata(fs, base, read_table=read_table)
        self.assertEqual(skipped, ["test"])
        self.assertEqual(len(fs.opened), 2 * len(W.CONFIGS))
        self.assertTrue(all(W.split_of(p) in ("dev", "train") for p in fs.opened + read))
        self.assertEqual({k[1] for k in files}, {"dev", "train"})
        self.assertEqual({r["split"] for rows in meta.values() for r in rows}, {"dev", "train"})
        self.assertIn("NOT re-verified", W.TEST_SPLIT_NOTE)

    def test_main_uses_the_skipping_reader(self):
        src = open(W.__file__, encoding="utf-8").read()
        main = src[src.index("def main():"):]
        self.assertIn("read_metadata(fs, base)", main)
        self.assertEqual(src.count("fs.open("), 2)          # read_parquet_columns + read_column (guarded)
        self.assertIn('if split not in OPENED_SPLITS:\n            raise RuntimeError', main)


# ------------------------------------------------------------------ end-to-end on synthetic tones
REF_RULE = (3, 2, 2.0)
S4_TEXT = "باب نهر جبل ورد بحر"


class FakeASR:
    """Deterministic fake: identifies the row by audio length and orig vs WORLD by exact equality.
    orig hypothesis drops the first reference word on even rows; WORLD drops the first two words."""

    def __init__(self, entries, faults):
        self.entries, self.faults, self.calls = entries, faults, collections.Counter()

    def make(self, asr):
        def fn(x):
            e = self.entries[len(x)]
            cond = "orig" if np.array_equal(x, e["x"]) else "world"
            self.calls[(asr, cond, e["i"])] += 1
            f = self.faults.get((asr, cond, e["i"]))
            if f == "always" or (f == "once" and self.calls[(asr, cond, e["i"])] == 1):
                raise RuntimeError("fake ASR fault")
            if f == "hang_once" and self.calls[(asr, cond, e["i"])] == 1:
                time.sleep(5)
            words = e["ref"].split()
            return " ".join(words[(1 if e["i"] % 2 == 0 else 0):] if cond == "orig" else words[2:])
        return fn


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_cls = W.load_world_class()

    def build(self):
        dev, text, audio, info = [], {}, {}, {}
        i = 0
        for s, spk in enumerate(["s1", "s2", "s3"]):
            for j in range(4):
                r = row(spk=spk, prompt=f"p{j % 3}", seg=f"g{j}", dur=1.0)
                words = [VOCAB[(i + m) % len(VOCAB)] for m in range(5)]
                t = " ".join(words) + (" <unintelligible>" if i == 3 else "") + (" <laugh>" if i == 5 else "")
                dev.append(r)
                text[W.row_key(r)] = t
                x = tone(0.7, f0=110 + 25 * s, extra=37 * i, seed=i)
                audio[W.row_key(r)] = wav_bytes(tone(0.7, f0=110, sr=22050, channels=2, extra=37 * i), sr=22050) \
                    if i == 1 else wav_bytes(x)
                info[W.row_key(r)] = dict(i=i, ref=" ".join(words))
                i += 1
        extra = [(row(spk="s1", seg="fail", dur=1.0), "كتاب بيت", RuntimeError("fetch")),   # source failure
                 (row(spk="s2", seg="long", dur=31.0), "كتاب", None),
                 (row(spk="s3", seg="tag", dur=1.0), "كتاب <music>", None),
                 (row(spk="s3", seg="notext", dur=1.0), None, None),
                 (row(cfg="apc_Arab", spk="s4", seg="g0", dur=1.0), S4_TEXT, wav_bytes(tone(0.6, extra=999)))]
        for r, t, a in extra:
            dev.append(r)
            if t is not None:
                text[W.row_key(r)] = t
            if a is not None:
                audio[W.row_key(r)] = a
        info[("apc_Arab", "s4", "p1", "g0")] = dict(i=12, ref=S4_TEXT)          # scored, not represented
        train = []
        for s, spk in enumerate(["t1", "t2", "t3"]):
            for j in range(5):
                r = row(spk=spk, prompt=f"q{j}", seg=f"h{j}", dur=4.0, split="train")
                train.append(r)
                silent = spk == "t3" and j < 3                                 # t3: only 2 voiced clips
                audio[W.row_key(r)] = wav_bytes(np.zeros(SR // 2, np.float32) if silent
                                                else tone(0.6, f0=100 + 40 * s, seed=100 + j))
        train.append(row(spk="t1", seg="short", dur=2.0, split="train"))
        dev_speakers = sorted({(r["config"], r["speaker_id"]) for r in dev})
        return dev, text, dev_speakers, train, audio, info

    def run_screen(self, workdir, faults=None, world_cls=None, factory=None):
        dev, text, dev_speakers, train, audio, info = self.build()
        fetched = []

        def fetch_audio(rows):
            fetched.extend(W.row_key(r) for r in rows)
            return {W.row_key(r): audio.get(W.row_key(r)) for r in rows}
        fake = {}

        def asr_factory():
            entries = {}
            for k, v in info.items():
                p = os.path.join(workdir, "src", hashlib.sha256(repr(k).encode()).hexdigest()[:24] + ".wav")
                if os.path.exists(p):
                    x, _ = sf.read(p, dtype="float32")
                    assert len(x) not in entries
                    entries[len(x)] = dict(v, x=x)
            fake["asr"] = FakeASR(entries, faults or {})
            return {a: fake["asr"].make(a) for a in W.ASR_NAMES}
        logs = []
        with mock.patch.object(W, "PRIMARY_RULE", REF_RULE), mock.patch.object(W, "GUARD_MIN_SPEAKERS", 3), \
                mock.patch.object(W, "GUARD_MIN_WORDS", 30):
            R = W.run_screen(dev, text, dev_speakers, train, fetch_audio, world_cls or self.world_cls,
                             factory or asr_factory, workdir, log=logs.append, B=500)
        return R, info, fake, fetched

    def expected(self, info, rows_i, world_fail_i, F_i, O_i):
        """Independent recomputation from the fake ASR's rule (5-word references)."""
        P = [i for i in rows_i if i not in O_i and i not in F_i]
        eo = {i: 1 if i % 2 == 0 else 0 for i in rows_i}
        ew = {i: 5 if i in world_fail_i else 2 for i in rows_i}
        n = 5 * len(P)
        wo, ww = sum(eo[i] for i in P) / n, sum(ew[i] for i in P) / n
        n2 = 5 * (len(P) + len(F_i))
        wo2 = (sum(eo[i] for i in P) + sum(eo[i] for i in F_i)) / n2
        ww2 = (sum(ew[i] for i in P) + 5 * len(F_i)) / n2
        return (wo, ww, (ww - wo) / wo), (wo2, ww2, (ww2 - wo2) / wo2)

    def test_full_pipeline(self):
        real = self.world_cls

        class FailingWorld(real):                                          # WORLD conversion failure on row 4
            def run(self, inp, out, seed, direction):
                if sf.info(inp).frames == int(0.7 * SR) + 37 * 4:          # row 4 (s2, g0): unique length
                    time.sleep(5)                                            # hung converter -> Timeout
                return real.run(self, inp, out, seed, direction)
        work = tempfile.mkdtemp()
        try:
            faults = {("whisper", "world", 6): "always",          # whisper F
                      ("whisper", "orig", 9): "hang_once",        # times out once, retry succeeds
                      ("mms", "orig", 7): "always",               # mms O
                      ("mms", "world", 8): "once"}                # retried, succeeds
            with mock.patch.object(W, "WORLD_TIMEOUT_S", 1), mock.patch.object(W, "ASR_TIMEOUT_S", 1):
                R, info, fake, fetched = self.run_screen(work, faults=faults, world_cls=FailingWorld)
            self.assertEqual(R["status"], "DONE")
            self.assertEqual(R["dev_filter"]["excluded"],
                             {"duration_over_30s": 1, "undocumented_tag": 1, "missing_reference": 1})
            self.assertEqual([f["key"][3] for f in R["source_failures"]], ["fail"])
            g = R["validity_guard_before_asr"]
            self.assertEqual((g["represented_speakers"], g["valid"]), (3, True))
            pop = R["population"]
            self.assertEqual((pop["train_speakers"], pop["retained_clips"]), (3, 8))
            self.assertEqual(pop["speakers_short"], [dict(speaker=["arz_Arab", "t3"], retained=2)])
            self.assertEqual(collections.Counter(r["status"] for r in pop["record"]),
                             {"retained": 8, "too_few_voiced_frames": 3})
            self.assertEqual([f["key"][1:] for f in R["world_conversion_failures"]], [["s2", "p0", "g0"]])
            self.assertEqual(fake["asr"].calls[("mms", "world", 8)], 2)
            mf = R["per_asr"]["mms"]["failures"]["asr_faulted_attempts"]
            self.assertEqual(sorted((x["condition"], x["failed"], len(x["attempt_faults"])) for x in mf),
                             [("orig", True, 2), ("world", False, 1)])          # row 7 failed; row 8 retried ok
            self.assertTrue(all(f == ["RuntimeError"] * len(f) for f in (x["attempt_faults"] for x in mf)))
            self.assertEqual(R["world_conversion_failures"][0]["check"], "Timeout")    # 600 s policy (here 1 s)
            self.assertEqual(R["world_timeouts"], 1)
            wf = R["per_asr"]["whisper"]["failures"]
            self.assertEqual(wf["asr_timeouts"], 1)                                   # row 9: timed out once, retried ok
            self.assertIn(dict(key=["arz_Arab", "s3", "p1", "g1"], condition="orig", attempt_faults=["Timeout"],
                               failed=False), wf["asr_faulted_attempts"])
            self.assertEqual(R["implementation_parameters"]["world_timeout_s"], 1)
            self.assertNotIn(("whisper", "world", 4), fake["asr"].calls)    # failed render never sent to ASR
            for a, F_i, O_i in (("whisper", [6], []), ("mms", [], [7])):
                s = R["per_asr"][a]
                exp_p, exp_w = self.expected(info, list(range(13)), [4], F_i, O_i)
                np.testing.assert_allclose([s["primary"]["wer_orig"], s["primary"]["wer_world"], s["primary"]["rel"]], exp_p)
                np.testing.assert_allclose([s["worst_case"]["wer_orig"], s["worst_case"]["wer_world"],
                                            s["worst_case"]["rel"]], exp_w)
                self.assertEqual(len(s["failures"]["asr_fail_world"]), len(F_i))
                self.assertEqual(len(s["failures"]["asr_fail_orig"]), len(O_i))
                self.assertTrue(s["guard_after_failures"]["valid"])
                self.assertIn("rel_undefined_resamples", s["bootstrap"])
                # §9 estimand: intervals resample every speaker in the primary set (s4 too), not only
                # the 3 represented ones, so they describe the same rows as the point estimate
                self.assertEqual(s["guard_after_failures"]["represented_speakers"], 3)
                self.assertEqual(s["bootstrap"]["n_speakers"], 4)
            self.assertIn(R["result"], ("PASS", "FAIL", "INCONCLUSIVE"))
            self.assertEqual(R["result"], W.decide(g, R["per_asr"])[0])
            dump = json.dumps(R, ensure_ascii=False, default=str)
            self.assertFalse(any("؀" <= ch <= "ۿ" for ch in dump), "Arabic text leaked into results")
            self.assertEqual(sorted(os.listdir(work)), ["pop", "src", "world"])  # no text/ dir is written
            self.assertNotIn(("arz_Arab", "t1", "p1", "short"), fetched)
        finally:
            W.cleanup([work], outdir=work, log=lambda *_: None)

    def test_scores_match_independent_recomputation(self):
        work = tempfile.mkdtemp()
        try:
            R, info, fake, _ = self.run_screen(work, faults={("whisper", "world", 6): "always",
                                                             ("mms", "orig", 7): "always"})
            for a, F_i, O_i in (("whisper", [6], []), ("mms", [], [7])):
                s = R["per_asr"][a]
                exp_p, exp_w = self.expected(info, list(range(13)), [], F_i, O_i)
                got_p = (s["primary"]["wer_orig"], s["primary"]["wer_world"], s["primary"]["rel"])
                got_w = (s["worst_case"]["wer_orig"], s["worst_case"]["wer_world"], s["worst_case"]["rel"])
                np.testing.assert_allclose(got_p, exp_p)
                np.testing.assert_allclose(got_w, exp_w)
                self.assertEqual(len(s["failures"]["asr_fail_orig"]), len(O_i))
                sens = s["sensitivity_without_unintelligible"]
                self.assertEqual(sens["rows"], s["primary"]["rows"] - 1)            # row 3 carries <unintelligible>
            self.assertEqual(R["tag_prevalence"]["<unintelligible>"]["rows"], 1)
            self.assertEqual(R["world_conversion_failures"], [])
            hs = R["hashes"]
            self.assertTrue(all(len(h["src_sha256"]) == 64 and len(h["world_sha256"]) == 64 for h in hs))
        finally:
            W.cleanup([work], outdir=work, log=lambda *_: None)

    def test_guard_failure_stops_before_population_and_asr(self):
        work = tempfile.mkdtemp()

        def factory():
            raise AssertionError("ASR must not load when the guard fails")
        try:
            dev, text, dev_speakers, train, audio, info = self.build()
            with mock.patch.object(W, "PRIMARY_RULE", REF_RULE), mock.patch.object(W, "GUARD_MIN_SPEAKERS", 4), \
                    mock.patch.object(W, "GUARD_MIN_WORDS", 30):
                R = W.run_screen(dev, text, dev_speakers, train,
                                 lambda rows: {W.row_key(r): audio.get(W.row_key(r)) for r in rows},
                                 self.world_cls, factory, work, log=lambda *_: None, B=10)
            self.assertEqual((R["status"], R["result"]), ("DONE", "INCONCLUSIVE"))
            self.assertNotIn("population", R)
            self.assertEqual(sorted(os.listdir(work)), ["src"])
        finally:
            W.cleanup([work], outdir=work, log=lambda *_: None)

    def test_population_hard_stop(self):
        work = tempfile.mkdtemp()
        try:
            dev, text, dev_speakers, train, audio, info = self.build()
            for r in train:
                audio[W.row_key(r)] = wav_bytes(np.zeros(SR // 2, np.float32))
            with mock.patch.object(W, "PRIMARY_RULE", REF_RULE), mock.patch.object(W, "GUARD_MIN_SPEAKERS", 3), \
                    mock.patch.object(W, "GUARD_MIN_WORDS", 30):
                R = W.run_screen(dev, text, dev_speakers, train,
                                 lambda rows: {W.row_key(r): audio.get(W.row_key(r)) for r in rows},
                                 self.world_cls, lambda: self.fail("no ASR after hard stop"), work,
                                 log=lambda *_: None, B=10)
            self.assertEqual((R["status"], R["result"]), ("STOPPED", "INCONCLUSIVE"))
            self.assertEqual(R["population"]["retained_clips"], 0)
        finally:
            W.cleanup([work], outdir=work, log=lambda *_: None)


class TestMainRefuses(unittest.TestCase):
    def test_requires_owner_approval(self):
        env = {k: v for k, v in os.environ.items() if k != "VOISE_SCREEN_APPROVED"}
        p = subprocess.run([sys.executable, os.path.join(HERE, "world_clarity_screen.py")], env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 1)
        self.assertIn("VOISE_SCREEN_APPROVED=1 is required", p.stderr)
        self.assertFalse(os.path.exists(W.WORK) and os.listdir(W.WORK))


if __name__ == "__main__":
    unittest.main()
