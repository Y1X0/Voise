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


if __name__ == "__main__":
    unittest.main()
