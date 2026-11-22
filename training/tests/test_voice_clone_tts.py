"""voice_clone_tts: the consent gate, output confinement and SYNTHETIC naming (no model is loaded)."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import voice_clone_tts as V  # noqa: E402

GOOD = {"voice_owner": "Test Person (as written)", "scope": "testing only", "revocable": "any time"}


class TestConsent(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def write(self, obj, name="c.json"):
        p = os.path.join(self.d, name)
        open(p, "w").write(obj if isinstance(obj, str) else json.dumps(obj))
        return p

    def test_missing_record_refused(self):
        with self.assertRaises(SystemExit) as c:
            V.load_consent(os.path.join(self.d, "nope.json"))
        self.assertIn("refusing", str(c.exception))

    def test_incomplete_or_invalid_record_refused(self):
        for bad in ({"voice_owner": "x"}, {"voice_owner": "x", "scope": "y"}, "{not json"):
            with self.assertRaises(SystemExit):
                V.load_consent(self.write(bad))

    def test_revoked_record_refused(self):
        with self.assertRaises(SystemExit) as c:
            V.load_consent(self.write(dict(GOOD, revoked=True)))
        self.assertIn("revoked", str(c.exception))

    def test_good_record_and_hash(self):
        rec, sha = V.load_consent(self.write(GOOD))
        self.assertEqual(rec["voice_owner"], GOOD["voice_owner"])
        self.assertEqual(len(sha), 64)


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self.runs = tempfile.mkdtemp()

    def test_outputs_confined_to_runs(self):
        with self.assertRaises(SystemExit):
            V.check_out_dir(tempfile.mkdtemp(), runs_root=self.runs)
        with self.assertRaises(SystemExit):
            V.check_out_dir(os.path.join(self.runs, "..", "escape"), runs_root=self.runs)
        out = V.check_out_dir(os.path.join(self.runs, "tts"), runs_root=self.runs)
        self.assertTrue(os.path.isdir(out))
        self.assertEqual(oct(os.stat(out).st_mode & 0o777), "0o700")

    def test_synthetic_naming_and_no_overwrite(self):
        out = V.check_out_dir(os.path.join(self.runs, "tts"), runs_root=self.runs)
        wav, meta = V.output_paths(out, "Test Person (as written)", 0, "20261025T000000Z")
        self.assertTrue(os.path.basename(wav).startswith("SYNTHETIC_test_person_"))
        self.assertTrue(meta.endswith(".json"))
        open(wav, "w").close()
        with self.assertRaises(SystemExit):
            V.output_paths(out, "Test Person (as written)", 0, "20261025T000000Z")


class TestTiming(unittest.TestCase):
    def test_short_sentence_gets_time(self):
        self.assertEqual(V.target_duration("أنا بحبك يا لول! إمتى أجي أشوفك، ونطلع؟"), 5.8)

    def test_long_sentence_or_explicit_speed_left_to_model(self):
        self.assertIsNone(V.target_duration(" ".join(["كلمة"] * 11)))
        self.assertIsNone(V.target_duration("جملة قصيرة", speed=0.9))


class TestLexicon(unittest.TestCase):
    def test_whole_words_only(self):
        lex = {"منحكي": "مِنْحكي"}
        self.assertEqual(V.apply_lexicon("بكرا منحكي، ومنحكي", lex), "بكرا مِنْحكي، ومنحكي")

    def test_notes_ignored_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lex.json")
            json.dump({"_note": "x", "ومنحكي": "ومِنْحكي"}, open(p, "w", encoding="utf-8"))
            self.assertEqual(V.load_lexicon(p), {"ومنحكي": "ومِنْحكي"})
            self.assertEqual(V.load_lexicon(os.path.join(d, "none.json")), {})

    def test_shipped_dictionary_loads(self):
        self.assertIsInstance(V.load_lexicon(), dict)


class TestTakeSelection(unittest.TestCase):
    """Engine.takes with a stand-in model and Whisper: what is heard is scripted per take."""

    def run_takes(self, heard, text="طب اسمع، بكرا بنطلع ومنحكي."):
        import numpy as np
        calls = []

        class Model:
            def generate(self, **kw):
                calls.append(kw)
                return np.zeros(2400, np.float32)

        e = V.Engine.__new__(V.Engine)
        e.model, e.language, e.temperature, e.sr, e.asr_dir, e.enc, e.mos = Model(), "apc", 0.5, 24000, None, None, None
        said = iter(heard)
        old = V.transcribe
        V.transcribe = lambda y, d: next(said)
        try:
            best, cands = e.takes({"prompt": None}, text, candidates=3)
        finally:
            V.transcribe = old
        return best, cands, calls

    def test_one_wrong_word_gets_another_take(self):
        best, cands, calls = self.run_takes(["طب اسمع، بكرا بنطلع وما نحكي.", "طب اسمع، بكرة بنطلع ومنحكي."])
        self.assertEqual(len(cands), 2)                 # «وما نحكي» was 3% of the letters but still rejected
        self.assertEqual(best["seed"], 1)

    def test_model_reads_dictionary_spelling_and_whisper_checks_it(self):
        import numpy as np
        calls = []

        class Model:
            def generate(self, **kw):
                calls.append(kw["text"])
                return np.zeros(2400, np.float32)

        e = V.Engine.__new__(V.Engine)
        e.model, e.language, e.temperature, e.sr, e.asr_dir, e.enc, e.mos = Model(), "apc", 0.5, 24000, None, None, None
        old = V.transcribe
        V.transcribe = lambda y, d: "وبنحكي"
        try:
            best, cands = e.takes({"prompt": None}, "ومنحكي", candidates=3, spoken="وبنحكي")
        finally:
            V.transcribe = old
        self.assertEqual(calls, ["وبنحكي"])
        self.assertEqual(best["cer"], 0.0)

    def test_wrong_vowel_gets_another_take(self):
        import numpy as np
        import types

        class Model:
            def generate(self, **kw):
                return np.zeros(2400, np.float32)

        e = V.Engine.__new__(V.Engine)
        e.model, e.language, e.temperature, e.sr, e.asr_dir, e.enc, e.mos = Model(), "apc", 0.5, 24000, None, None, None
        verdicts = iter([[("بكره", "u-a", "i")], []])          # first take says bikra, second bukra
        e.check_vowels = True
        e.pronlex_mod = types.SimpleNamespace(load=lambda: {"بكره": {"vowels": "u-a"}},
                                              check=lambda y, t, lex, d: next(verdicts))
        old = V.transcribe
        V.transcribe = lambda y, d: "بكرا منحكي"
        try:
            best, cands = e.takes({"prompt": None}, "بكرا منحكي", candidates=3)
        finally:
            V.transcribe = old
        self.assertEqual(len(cands), 2)                    # Whisper heard every word both times
        self.assertEqual(best["seed"], 1)

    def test_timing_from_model_and_last_take_slower(self):
        _, cands, calls = self.run_takes(["خطأ"] * 3)
        self.assertEqual(len(cands), 3)
        self.assertTrue(all("duration" not in c or c["duration"] is None for c in calls))
        self.assertEqual([c["speed"] for c in calls], [None, None, 0.85])


class TestTakeTarget(unittest.TestCase):
    """take_target "person": takes are ranked by closeness to the person's voice centre, every take is made, and
    takes from extra references compete with the main reference's (experiments/take_select_2026-11-17)."""

    def engine(self, take_target, embs):
        import numpy as np
        import torch

        class Model:
            def generate(self, **kw):
                return np.zeros(2400, np.float32)

        e = V.Engine.__new__(V.Engine)
        e.model, e.language, e.temperature, e.sr, e.asr_dir, e.mos = Model(), "apc", 0.5, 24000, None, None
        e.enc, e.take_target = object(), take_target
        it = iter([torch.tensor(x, dtype=torch.float32) for x in embs])
        self.old = V.speaker_embedding, V.transcribe
        V.speaker_embedding = lambda enc, y: next(it)
        V.transcribe = lambda y, d: "بكرا منحكي"
        return e

    def tearDown(self):
        if hasattr(self, "old"):
            V.speaker_embedding, V.transcribe = self.old

    def voice(self, ref_emb, strategy="quality", centroid=(1.0, 0.0)):
        import torch
        return dict(prompt=strategy, emb=torch.tensor(ref_emb, dtype=torch.float32), centroid=list(centroid),
                    strategy=strategy)

    def test_default_measures_against_reference_and_stops_at_first_clean_take(self):
        e = self.engine("reference", [(0.0, 1.0)])
        best, cands = e.takes(self.voice((0.0, 1.0)), "بكرا منحكي", candidates=3)
        self.assertEqual(len(cands), 1)                     # unchanged behaviour: a clean first take ends it
        self.assertEqual(best["sim"], 1.0)                  # similarity to the reference clip, not the centre

    def test_person_target_makes_every_take_and_picks_closest_to_centre(self):
        e = self.engine("person", [(0.0, 1.0), (0.8, 0.6), (0.6, 0.8)])
        best, cands = e.takes(self.voice((0.0, 1.0)), "بكرا منحكي", candidates=3)
        self.assertEqual(len(cands), 3)
        self.assertEqual(best["seed"], 1)                   # 0.8 to the centre (1, 0); seed 0 matched the reference
        self.assertEqual([c["sim"] for c in cands], [0.0, 0.8, 0.6])

    def test_extra_reference_takes_compete_with_same_seeds(self):
        e = self.engine("person", [(0.6, 0.8), (0.7, 0.71), (0.9, 0.44), (0.5, 0.87)])
        best, cands = e.takes(self.voice((0.0, 1.0)), "بكرا منحكي", candidates=2,
                              extra=[self.voice((1.0, 0.0), "similarity_first")])
        self.assertEqual([(c["reference"], c["seed"]) for c in cands],
                         [("quality", 0), ("quality", 1), ("similarity_first", 0), ("similarity_first", 1)])
        self.assertEqual((best["reference"], best["seed"]), ("similarity_first", 0))

    def test_words_still_come_first(self):
        e = self.engine("person", [(1.0, 0.0), (0.5, 0.87)])
        said = iter(["خطأ كبير", "بكرا منحكي"])
        V.transcribe = lambda y, d: next(said)
        best, _ = e.takes(self.voice((0.0, 1.0)), "بكرا منحكي", candidates=2)
        self.assertEqual(best["seed"], 1)                   # the take closest to the person misread the words

    def test_without_centre_falls_back_to_reference(self):
        e = self.engine("person", [(0.0, 1.0)])
        v = self.voice((0.0, 1.0))
        v["centroid"] = None
        best, cands = e.takes(v, "بكرا منحكي", candidates=3)
        self.assertEqual((len(cands), best["sim"]), (1, 1.0))

    def test_unknown_target_or_pool_refused(self):
        with self.assertRaises(SystemExit):
            V.take_target_vector({}, "nearest")
        import inspect
        d = inspect.signature(V.Engine.__init__).parameters
        self.assertEqual((d["take_target"].default, d["reference_pool"].default), ("person", ("similarity_first",)))


class TestSplitText(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        self.assertEqual(V.split_text("شو أخبارك؟ إن شاء الله منيح."), ["شو أخبارك؟ إن شاء الله منيح."])

    def test_paragraph_splits_at_sentences_and_keeps_every_word(self):
        para = ("اليوم صحيت بكير وشربت قهوة وطلعت عالجامعة بس الطريق كان زحمة كثير. "
                "لما وصلت لقيت المحاضرة تأجلت، فقعدت مع صحابي بالكافتيريا وحكينا عن المشروع ساعتين! "
                "بعدين رجعت عالبيت تعبان.")
        pieces = V.split_text(para, max_words=12)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p.split()) <= 12 for p in pieces))
        self.assertEqual(" ".join(pieces).split(), para.split())

    def test_reaction_tag_stays_with_its_words(self):
        self.assertIn("[laughter] ولك", " ".join(V.split_text("[laughter] ولك والله ما بصدق!")))


class TestReactions(unittest.TestCase):
    def test_reaction_tags_are_recognised(self):
        self.assertTrue(V.REACTION.search("[laughter] يلا خلص"))
        self.assertTrue(V.REACTION.search("شو؟ [surprise-wa]"))
        self.assertFalse(V.REACTION.search("يلا خلص [not-a-tag]"))


class TestTakes(unittest.TestCase):
    def test_normalize_ignores_tags_marks_and_spelling_variants(self):
        self.assertEqual(V.normalize_ar("[laughter] أهلين، كيفكِ؟"), V.normalize_ar("اهلين كيفك"))

    def test_dialect_spelling_variants_are_not_mistakes(self):
        self.assertEqual(V.cer("شو رأيك نطلع بكرا عالبلد؟", "شو رأيك نطلع بكرة ع البلد؟"), 0.0)
        self.assertGreater(V.cer("يلا خلص", "إلا خلاص"), 0.05)

    def test_cer(self):
        self.assertEqual(V.cer("إمتى صار هاد", "امتى صار هاد"), 0.0)
        self.assertGreater(V.cer("إمتى صار هاد", "انتصار هاد"), 0.1)

    def test_choose_best_prefers_words_then_voice(self):
        cands = [dict(seed=0, cer=0.20, sim=0.9), dict(seed=1, cer=0.00, sim=0.4), dict(seed=2, cer=0.01, sim=0.6)]
        self.assertEqual(V.choose_best(cands)["seed"], 2)
        self.assertEqual(V.choose_best([dict(seed=0, cer=0.1, sim=None), dict(seed=1, cer=0.0, sim=None)])["seed"], 1)



class TestVersionPerVoice(unittest.TestCase):
    """Each voice uses the version that keeps it most distinct (voice_versions.json from voise_learn.py choose)."""

    def test_choice_file_for_another_generation_is_ignored(self):
        import tempfile
        d = tempfile.mkdtemp()
        f = os.path.join(d, V.VERSIONS_FILE)
        json.dump({"adapter_generation": 2, "voices": {"omar": "base", "x": "adapter", "y": "junk"}}, open(f, "w"))
        self.assertEqual(V.load_versions(f, 2), {"omar": "base", "x": "adapter"})
        self.assertEqual(V.load_versions(f, 3), {})               # a new generation must be measured again
        self.assertEqual(V.load_versions(os.path.join(d, "none.json"), 2), {})

    def test_swaps_only_when_the_person_needs_another_version(self):
        loaded = []

        class Model:
            def load_state_dict(self, w, strict):
                loaded.append(w["k"])

        e = V.Engine.__new__(V.Engine)
        e.model, e.in_use = Model(), "adapter"
        e.weights = {"base": {"k": "base"}, "adapter": {"k": "adapter"}}
        e.versions = {"omar": "base"}
        self.assertEqual(e.use_version_for("runs/learn/pool/omar/"), "base")
        self.assertEqual(e.use_version_for("runs/learn/pool/omar"), "base")
        self.assertEqual(e.use_version_for("runs/learn/pool/mafia_abood"), "adapter")   # not listed: learned one
        self.assertEqual(loaded, ["base", "adapter"])

    def test_without_learned_version_it_is_always_base(self):
        e = V.Engine.__new__(V.Engine)
        e.weights, e.versions, e.in_use = {}, {"omar": "adapter"}, "base"
        self.assertEqual(e.use_version_for("pool/omar"), "base")



class TestWatermarkFailsClosed(unittest.TestCase):
    def test_no_unmarked_output_when_watermark_unavailable(self):
        import builtins
        import numpy as np
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "perth":
                raise ImportError("perth missing")
            return real_import(name, *a, **k)
        builtins.__import__ = fake_import
        try:
            with self.assertRaises(SystemExit):
                V.watermark(np.zeros(1600, np.float32), 16000)
        finally:
            builtins.__import__ = real_import

    def test_no_bypass_flag(self):
        src = open(V.__file__, encoding="utf-8").read()
        self.assertNotIn("--no-watermark", src)


class TestPolish(unittest.TestCase):
    """The automatic voice-conversion polish, with stand-ins for the encoder, the converter and Whisper."""

    def engine(self, sims, heard):
        import numpy as np
        import torch
        e = V.Engine.__new__(V.Engine)
        e.sr, e.asr_dir, e.enc = 16000, None, object()
        it = iter(sims)
        self._old = (V.speaker_embedding, V.transcribe)
        V.speaker_embedding = lambda enc, y: torch.tensor([next(it), 0.0])
        V.transcribe = lambda y, d: heard

        class Conv:
            def convert(self, y, d):
                return np.zeros(1600, np.float32), 60.0
        e.converter = Conv()
        return e

    def tearDown(self):
        V.speaker_embedding, V.transcribe = self._old

    def run_polish(self, sims, heard, cer_before=0.0):
        import numpy as np
        e = self.engine(sims, heard)
        v = {"centroid": [1.0, 0.0], "ceiling": 0.8}
        best = {"y": np.zeros(1600, np.float32), "cer": cer_before}
        return e.polish(v, "dir", "والله يا جماعة", best)

    def test_close_take_is_left_alone(self):
        self.assertIsNone(self.run_polish([0.70], "والله يا جماعة"))          # 0.70/0.8 = 88% >= 80%

    def test_far_take_is_polished_when_closer_and_words_hold(self):
        r = self.run_polish([0.40, 0.72], "والله يا جماعة")                  # 50% -> 90%, same words
        self.assertTrue(r["polished"])
        self.assertEqual(r["polish_closeness_before"], 0.5)

    def test_polish_rejected_when_it_garbles_words(self):
        r = self.run_polish([0.40, 0.72], "والله جماعة كلام ثاني خالص")
        self.assertFalse(r["polished"])
        self.assertNotIn("y", r)


class TestSimilarityFirst(unittest.TestCase):
    """Experimental reference strategy (step 3b): most similar clips first, each only if Whisper hears it clearly."""
    WORDS = "كان في صوتين وأنا قلت"

    def cands(self, *rows):
        return [dict(name=n, sim=s, dur=d) for n, s, d in rows]

    def heard(self, clarity=None, text=None):
        calls = []

        def fn(c):
            calls.append(c["name"])
            return (text or {}).get(c["name"], self.WORDS), (clarity or {}).get(c["name"], 0.85)
        return fn, calls

    def test_most_similar_first_within_19_s(self):
        fn, _ = self.heard()
        chosen, rejected = V.pick_similar(self.cands(("c", 0.70, 6), ("a", 0.90, 8), ("b", 0.80, 8), ("d", 0.60, 2)), fn)
        self.assertEqual([c["name"] for c in chosen], ["a", "b", "d"])      # c (6 s) no longer fits; d (2 s) does
        self.assertLessEqual(sum(c["dur"] for c in chosen) + 0.3 * (len(chosen) - 1), 19.0)
        self.assertEqual(rejected, [])

    def test_similar_clip_with_unclear_text_never_goes_in(self):
        fn, _ = self.heard(clarity={"a": 0.55})
        chosen, rejected = V.pick_similar(self.cands(("a", 0.95, 5), ("b", 0.80, 5)), fn)
        self.assertEqual([c["name"] for c in chosen], ["b"])
        self.assertEqual([(r["name"], r["reason"]) for r in rejected], [("a", "unclear")])

    def test_unknown_clarity_and_empty_text_rejected(self):
        fn, _ = self.heard(clarity={"a": None}, text={"b": "اه"})
        chosen, rejected = V.pick_similar(self.cands(("a", 0.95, 5), ("b", 0.90, 5), ("c", 0.80, 5)), fn)
        self.assertEqual([c["name"] for c in chosen], ["c"])
        self.assertEqual({r["name"]: r["reason"] for r in rejected}, {"a": "unclear", "b": "no words"})

    def test_threshold_is_inclusive_and_configurable(self):
        fn, _ = self.heard(clarity={"a": V.MIN_CLARITY})
        self.assertEqual(len(V.pick_similar(self.cands(("a", 0.9, 5)), fn)[0]), 1)
        self.assertEqual(len(V.pick_similar(self.cands(("a", 0.9, 5)), fn, min_clarity=0.9)[0]), 0)

    def test_whisper_runs_only_on_clips_that_fit_and_stops_when_full(self):
        fn, calls = self.heard()
        V.pick_similar(self.cands(("a", 0.9, 18.5), ("big", 0.8, 5), ("b", 0.7, 0.5), ("c", 0.6, 0.4)), fn)
        self.assertEqual(calls, ["a"])                           # full after a: < 1 s left

    def test_nothing_clear_gives_nothing(self):
        fn, _ = self.heard(clarity={"a": 0.2, "b": 0.3})
        chosen, rejected = V.pick_similar(self.cands(("a", 0.9, 5), ("b", 0.8, 5)), fn)
        self.assertEqual(chosen, [])
        self.assertEqual(len(rejected), 2)

    def test_similarity_is_cosine_to_centre_of_first_30_clips(self):
        import torch
        n = torch.nn.functional.normalize
        own = [n(torch.tensor([1.0, 0.1 * i, 0.0]), dim=0) for i in range(30)]
        later = n(torch.tensor([0.0, 0.0, 1.0]), dim=0)            # 31st clip: not part of the centre
        sims = V.similarity_to_person(own + [later])
        centre = n(sum(own), dim=0)
        self.assertAlmostEqual(sims[0], float(own[0] @ centre), places=5)
        self.assertAlmostEqual(sims[-1], 0.0, places=5)

    def test_strategy_names_and_default(self):
        import inspect
        self.assertEqual(V.REFERENCE_STRATEGIES, ("quality", "similarity_first"))
        self.assertEqual(inspect.signature(V.select_reference).parameters["strategy"].default, "quality")
        self.assertEqual(inspect.signature(V.Engine.__init__).parameters["reference_strategy"].default, "quality")
        with self.assertRaises(SystemExit):
            V.select_reference(tempfile.mkdtemp(), "nowhere", strategy="diverse")

    def test_without_speaker_encoder_falls_back_to_quality(self):
        seen = {}
        orig = V.select_reference
        try:
            V.select_reference = lambda d, asr, enc, long_s=19.0, **k: seen.setdefault("args", (d, enc, long_s)) and ("p", None, "t", {})
            V.select_reference_similarity_first("dir", "asr", None)
        finally:
            V.select_reference = orig
        self.assertEqual(seen["args"], ("dir", None, 19.0))


if __name__ == "__main__":
    unittest.main()
