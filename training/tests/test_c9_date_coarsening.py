"""C9 / R10 regression tests: exported manifests carry no participant timing finer than a month
(docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md R10; datasets/build_manifests.py
participant_timing_checks).

Synthetic only: tiny generated WAV tones, fake pseudonyms and consent ids, temporary directories.
No real audio, no participants, no network.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
from datasets import build_manifests as BM  # noqa: E402
from datasets import consent as CO  # noqa: E402

FULL_DATE = re.compile(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])")
FINE_VALUES = {                                   # value -> kind it must be reported as
    "2026-10-11": "date", "20261011": "date", "11-10-2026": "date", "2026/10/11": "date",
    "2026_10_1": "date", "2026-10-11T14:03:22Z": "date", "Oct11": "date", "11oct2026": "date",
    "october 11": "date", "11-Sept": "date",
    "14:03": "time", "14h03": "time", "14-03-22": "date|time",     # ambiguous: 14:03:22 or 14 Mar 2022
    "1760191402": "epoch", "1760191402000": "epoch",
    # Arabic-Indic / Eastern Arabic-Indic digits, Arabic and Hijri month names
    "٢٠٢٦-١٠-١١": "date", "۲۰۲۶/۱۰/۱۱": "date", "٢٠٢٦١٠١١": "date", "١٤:٠٣": "time",
    "11 أكتوبر": "date", "أكتوبر 11": "date", "١١ تشرين الأول": "date", "١٠ ربيع الآخر ١٤٤٨": "date",
    "1448-04-10": "date",
    # ordinal (day-of-year) and ISO week dates, two-digit years, compact day-month-year
    "2026-284": "date", "2026284": "date", "2026-W41": "date", "2026w41": "date",
    "11-10-26": "date|time", "11/10/26": "date", "11102026": "date",
}
ALLOWED_VALUES = ("2026-10", "trial-2026-10", "pilot-2026-11", "day1", "p1.ogg", "marker3", "decoder2", "augment-2",
                  "own-ar-v1", "٢٠٢٦-١٠", "أكتوبر 2026", "تشرين الأول", "v1.2.26", "part-284", "build-100", "2026",
                  "prompt-12")


def write_wav(path, seconds=0.3, freq=440):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(path, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 16000)


def consent_record(cid, pseud, withdrawn=False):
    """Schema-valid synthetic record (same scopes as test_data_pipeline.consent_record) with a
    precise consent time, so the tests can show that it never reaches a manifest."""
    sc = {"ml_training": True, "voice_anonymization_rnd": True, "commercial_product_development": True,
          "model_evaluation": True, "derivative_model_training": True, "audio_redistribution": False}
    return {"consent_id": cid, "speaker_pseudonym": pseud, "form_version": "consent-ar-en-v1.0",
            "form_sha256": "0" * 64, "consent_timestamp": "2026-10-01T14:03:22Z", "adult_confirmed": True,
            "language": ["ar"], "dialect_primary": "Jordanian", "scopes": sc,
            "withdrawal": {"status": "withdrawn" if withdrawn else "active",
                           "withdrawn_at": "2026-10-02T09:00:00Z" if withdrawn else None},
            "retention_until": "2028-10-01", "withdrawal_token_sha256": "a" * 64, "collector": "synthetic-test"}


def own_tree(root, sessions=("ses-0a1b2c3d", "ses-1b2c3d4e"), n_spk=4, fname="p1.wav"):
    for s in range(n_spk):
        spk = f"spk-{s:010x}"
        for i, sess in enumerate(sessions):
            write_wav(os.path.join(root, spk, sess, fname), 0.3, 300 + 40 * s + i)
        with open(os.path.join(root, spk, "consent_id.txt"), "w") as f:
            f.write(f"CNS-{s:012x}")
    return root


def participant_row(**kw):
    r = {"path": "/store/spk-0123456789/ses-0a1b2c3d/p1.wav", "speaker": "own_recordings:spk-0123456789",
         "session": "ses-0a1b2c3d", "corpus": "own_recordings", "subset": "all", "language": "ar",
         "duration": 1.0, "sr": 16000, "text": "نص", "style": "conversational", "consent_id": "CNS-0123456789ab",
         "split": "train"}
    r.update(kw)
    return r


class Detector(unittest.TestCase):
    def test_finer_than_month_values_detected(self):
        for v, kind in FINE_VALUES.items():
            with self.subTest(v=v):
                self.assertIn(BM.finer_than_month(v), kind.split("|"))

    def test_month_precision_and_ordinary_names_allowed(self):
        for v in ALLOWED_VALUES:
            with self.subTest(v=v):
                self.assertIsNone(BM.finer_than_month(v))


class ParticipantRows(unittest.TestCase):
    def check(self, rows, inputs=None):
        return BM.participant_timing_checks(rows, inputs)

    def test_compliant_rows_pass(self):
        rows = [participant_row(), participant_row(subset="trial-2026-10"),
                participant_row(sha256="ab" * 32, path="/o/audio/spk-0123456789/ses-0a1b2c3d/rec-2026101114032200.flac")]
        self.assertEqual(self.check(rows), [])

    def test_random_ids_that_look_like_dates_are_not_refused(self):
        # schema-format random ids are masked: an all-digit hex id is not a date
        rows = [participant_row(session="ses-20261011", path="/s/spk-1760191402/ses-20261011/rec-2026101114032200.flac",
                                speaker="own_recordings:spk-1760191402", sha256="2026101114" + "0" * 54)]
        self.assertEqual(self.check(rows), [])

    def test_dated_session_path_subset_refused(self):
        for field in ("subset", "path", "style", "device", "dialect", "role"):
            for v, kind in FINE_VALUES.items():
                value = f"/store/spk-0123456789/{v}/p1.wav" if field == "path" else v
                with self.subTest(field=field, v=v):
                    errs = self.check([participant_row(**{field: value})])
                    self.assertEqual(len(errs), 1, errs)
                    self.assertIn(repr(field), errs[0])
                    self.assertTrue(any(k in errs[0] for k in kind.split("|")), errs[0])
                    self.assertNotIn(v, errs[0])                # error messages never echo the value

    def test_ids_must_have_their_random_schema_form(self):
        cases = {"session": ("day1", "2026-10-11", "trial-2026-10", "ses-0A1B2C3D", "ses-0a1b2c3", "ses-0a1b2c3d4",
                             "SES-0a1b2c3d", "ses-٠a1b2c3d", ""),
                 "speaker": ("own_recordings:2026-10-11", "own_recordings:spk-012345678", "own_recordings:ahmad"),
                 "consent_id": ("CNS-2026-10-11", "CNS-0123456789AB", "CNS-0123456789a")}
        for field, values in cases.items():
            for v in values:
                with self.subTest(field=field, v=v):
                    errs = self.check([participant_row(**{field: v})])
                    self.assertTrue(any(f"participant {field} is not a random schema id" in e for e in errs), errs)
                    if v:
                        self.assertFalse(any(v in e for e in errs))     # never echoed
        r = participant_row()
        del r["session"]
        self.assertTrue(self.check([r]))

    def test_dates_next_to_random_ids_are_still_found(self):
        """Only exact schema-form ids are masked: a date attached to or shaped like an id is caught."""
        for p in ("/s/spk-0123456789/ses-0a1b2c3d/rec-2026-10-11.wav",
                  "/s/spk-0123456789/ses-0a1b2c3d/ses-0a1b2c3d_2026-10-11.wav",
                  "/s/spk-0123456789/ses-0a1b2c3d/spk-0123456789-14:03.wav"):
            with self.subTest(p=p):
                self.assertTrue(self.check([participant_row(path=p)]))

    def test_inputs_with_arabic_indic_digits_refused(self):
        self.assertTrue(self.check([participant_row()], {"consent_store": "/secure/consents-٢٠٢٦-١٠-١١.jsonl"}))

    def test_h_time_form_not_matched_inside_identifiers(self):
        for v in ("/data/tmp1h23ab/x.wav", "/data/ch12h30x/x.wav", "build-v2h10"):
            with self.subTest(v=v):
                self.assertIsNone(BM.finer_than_month(v))
        self.assertEqual(BM.finer_than_month("rec_14h03"), "time")

    def test_timing_check_leaves_text_to_the_text_check(self):
        """The timing check does not scan `text` (speech content); the separate text check does."""
        row = participant_row(text="سجلت هذا يوم 2026-10-11 الساعة 14:03")
        self.assertEqual(self.check([row]), [])
        self.assertIn("date_or_time", BM.text_privacy_indicators(row["text"]))   # 8-digit dates may also read as numbers

    def test_dated_root_or_file_name_refused(self):
        for p in ("/data/2026-10-11/spk-0123456789/ses-0a1b2c3d/p1.wav",
                  "/data/own/spk-0123456789/ses-0a1b2c3d/voice_20261011_140322.wav",
                  "/data/own/spk-0123456789/ses-0a1b2c3d/audio_1760191402.ogg",
                  "/data/own/spk-0123456789/ses-0a1b2c3d/تسجيل_١١_أكتوبر.ogg",
                  "/data/٢٠٢٦-١٠-١١/spk-0123456789/ses-0a1b2c3d/p1.wav",
                  "/data/own/spk-0123456789/ses-0a1b2c3d/day-2026-284.wav"):
            with self.subTest(p=p):
                self.assertTrue(self.check([participant_row(path=p)]))

    def test_extra_fields_refused(self):
        for k in ("recorded_at", "consent_timestamp", "withdrawn_at", "retention_until", "timestamp", "created_at",
                  "mtime", "recorded_month"):
            with self.subTest(k=k):
                errs = self.check([participant_row(**{k: "2026-10"})])
                self.assertTrue(any(k in e for e in errs), errs)

    def test_bad_sha256_refused(self):
        for v in ("2026-10-11", "xyz", "ab" * 31):
            with self.subTest(v=v):
                self.assertTrue(self.check([participant_row(sha256=v)]))

    def test_any_row_with_consent_id_is_participant(self):
        self.assertTrue(self.check([participant_row(corpus="renamed_corpus", speaker="renamed_corpus:x",
                                                    session="2026-10-11")]))

    def test_public_corpora_not_checked(self):
        row = {"path": "/vp/20090114-0900-PLENARY-3.wav", "speaker": "voxpopuli_en:abc",
               "session": "20090114-0900-PLENARY", "corpus": "voxpopuli_en", "language": "en", "duration": 2.0,
               "sr": 16000, "split": "train", "recorded_at": "2009-01-14"}
        self.assertEqual(self.check([row]), [])

    def test_inputs_checked_only_with_participant_rows(self):
        dated = {"consent_store": "/secure/consents-2026-10-11.jsonl"}
        self.assertTrue(self.check([participant_row()], dated))
        self.assertEqual(self.check([participant_row()], {"consent_store": "/secure/consents.jsonl"}), [])
        public = dict(participant_row(), corpus="mls_en", speaker="mls_en:1")
        public.pop("consent_id")
        self.assertEqual(self.check([public], dated), [])


TEXT_POSITIVE = {                                   # text -> indicator kind that must be reported
    "اتصل بي على ٠٧٩١٢٣٤٥٦٧": "phone_or_id_number", "call me on +962 79 123 4567": "phone_or_id_number",
    "my email is a.b@example.com": "email", "تابعني على @someone_99": "handle", "t.me/someone": "telegram_link",
    "زوروا www.example.com": "url", "سجلت هذا يوم ١١ أكتوبر": "date_or_time", "the meeting is at 14:03": "date_or_time",
    "اسمي أحمد": "self_introduction", "my name is Sara": "self_introduction", "الأستاذ خالد قال": "title_with_name",
    "Dr. Smith said": "title_with_name", "أبو محمد وصل": "kunya", "أسكن في شارع الجامعة 12": "address",
    "building 4, apt 7": "address", "user id: 123456": "id_number",
    # obfuscated forms: zero-width characters inside a number, full-width @ (NFKC)
    "رقمي 079\u200b123\u200d4567": "phone_or_id_number", "a.b＠example.com": "email",
}
TEXT_NEGATIVE = ("مرحبا كيف حالك اليوم يا صديقي", "الجو جميل في عام ٢٠٢٦", "هل تريد شاي أم قهوة", "أحب القهوة في الصباح",
                 "the quick brown fox jumps", "I read 3 books in October", "في الصف 12 طالبا", "حي على الصلاة",
                 "سأذهب إلى السوق غدا", "اسم الكتاب طويل", "")


KIND_WORDS = {"email", "url", "handle", "address", "kunya", "phone", "number", "time", "date"}


class TextIndicators(unittest.TestCase):
    """Separate personal-data indicator check on participant text (C9). Pattern-based: a hit means
    human review; no hit is NOT a guarantee that the text holds no personal data."""

    def test_positive_cases_flagged_with_kind(self):
        for text, kind in TEXT_POSITIVE.items():
            with self.subTest(text=text):
                self.assertIn(kind, BM.text_privacy_indicators(text))

    def test_ordinary_text_not_flagged(self):
        for text in TEXT_NEGATIVE:
            with self.subTest(text=text):
                self.assertEqual(BM.text_privacy_indicators(text), [])

    def test_rows_refused_and_messages_never_echo_text(self):
        for text in TEXT_POSITIVE:
            with self.subTest(text=text):
                errs = BM.participant_text_checks([participant_row(text=text)])
                self.assertEqual(len(errs), 1)
                self.assertIn("needs human review", errs[0])
                for token in set(re.findall(r"[\w@.+]{3,}", text)) - KIND_WORDS:
                    self.assertNotIn(token, errs[0])
                with self.assertRaises(BM.LeakageError) as cm:
                    BM.assert_no_leakage([participant_row(text=text, split="train")])
                for token in set(re.findall(r"[\w@.+]{3,}", text)) - KIND_WORDS:
                    self.assertNotIn(token, str(cm.exception))

    def test_public_corpora_text_not_checked(self):
        row = {"path": "/l/1.flac", "speaker": "mls_en:1", "session": "c1", "corpus": "mls_en", "language": "en",
               "duration": 2.0, "sr": 16000, "split": "train", "text": "my name is Sara, call 0791234567"}
        self.assertEqual(BM.participant_text_checks([row]), [])


class BuilderEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def store(self, n=4, withdrawn=()):
        p = os.path.join(self.tmp, "consents.jsonl")
        with open(p, "w") as f:
            for s in range(n):
                cid = f"CNS-{s:012x}"
                f.write(json.dumps(consent_record(cid, f"own_recordings:spk-{s:010x}", withdrawn=s in withdrawn)) + "\n")
        return p

    def cli(self, generic, out, store):
        script = os.path.join(TRAINING, "datasets", "build_manifests.py")
        return subprocess.run([sys.executable, script, "--out", out, "--seed", "5", "--licence-path", "research",
                               "--consent-store", store, "--generic", f"{generic}:own_recordings:ar"],
                              capture_output=True, text=True)

    def test_compliant_build_has_no_fine_dates_and_keeps_links(self):
        g = own_tree(os.path.join(self.tmp, "own"))
        out = os.path.join(self.tmp, "m")
        r = self.cli(g, out, self.store())
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = []
        for name in sorted(os.listdir(out)):
            with open(os.path.join(out, name), encoding="utf-8") as f:
                blob = f.read()
            self.assertIsNone(FULL_DATE.search(blob), name)      # consent store has 2026-10-01T14:03:22Z
            self.assertNotIn("consent_timestamp", blob)
            self.assertNotIn("14:03", blob)
            if name.endswith(".jsonl"):
                rows += [json.loads(l) for l in blob.splitlines() if l.strip()]
        self.assertEqual(len(rows), 8)
        for row in rows:                                        # values unchanged: lineage / withdrawal intact
            self.assertTrue(os.path.exists(row["path"]))
            spk = row["speaker"].split(":")[1]
            self.assertEqual(row["path"], os.path.join(g, spk, row["session"], "p1.wav"))
            self.assertEqual(row["consent_id"], f"CNS-{int(spk[4:], 16):012x}")
        BM.verify(out)

    def test_non_random_session_directory_refused(self):
        """No exception for test data: the old fixture names (day1/day2) are refused like any other."""
        g = own_tree(os.path.join(self.tmp, "own"), sessions=("day1", "day2"))
        r = self.cli(g, os.path.join(self.tmp, "m"), self.store())
        self.assertEqual(r.returncode, 1)
        self.assertIn("participant session is not a random schema id", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "m", "train.jsonl")))

    def test_dated_session_directory_refused_before_writing(self):
        g = own_tree(os.path.join(self.tmp, "own"), sessions=("2026-10-11", "2026-10-12"))
        out = os.path.join(self.tmp, "m")
        r = self.cli(g, out, self.store())
        self.assertEqual(r.returncode, 1)
        self.assertIn("finer than a month", r.stderr)
        self.assertNotIn("2026-10-11", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "train.jsonl")))

    def test_dated_file_name_refused(self):
        g = own_tree(os.path.join(self.tmp, "own"), fname="voice_20261011_140322.wav")
        r = self.cli(g, os.path.join(self.tmp, "m"), self.store())
        self.assertEqual(r.returncode, 1)
        self.assertIn("'path'", r.stderr)

    def test_write_refuses_dated_inputs(self):
        g = own_tree(os.path.join(self.tmp, "own"))
        rows, dropped = BM.assign_splits(BM.build_generic(g, "own_recordings", "ar"), 5, fractions=(0.5, 0.25, 0.25))
        with self.assertRaises(BM.LeakageError):
            BM.write(rows, os.path.join(self.tmp, "m"), 5, {"consent_store": "/s/consents-2026-10-11.jsonl"}, dropped)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "m", "train.jsonl")))

    def test_verify_refuses_fine_date_added_after_build(self):
        g = own_tree(os.path.join(self.tmp, "own"))
        out = os.path.join(self.tmp, "m")
        self.assertEqual(self.cli(g, out, self.store()).returncode, 0)
        p = os.path.join(out, "train.jsonl")
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        rows[0]["recorded_at"] = "2026-10-11"                   # edited after build, index regenerated
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        idx_p = os.path.join(out, "manifest_index.json")
        idx = json.load(open(idx_p))
        idx["manifests"]["train"] = BM._sha(p)
        json.dump(idx, open(idx_p, "w"))
        with self.assertRaises(BM.LeakageError):
            BM.verify(out)

    def test_transcript_with_indicator_refused_without_echo(self):
        g = own_tree(os.path.join(self.tmp, "own"))
        with open(os.path.join(g, "spk-0000000001", "ses-0a1b2c3d", "p1.txt"), "w", encoding="utf-8") as f:
            f.write("اسمي سارة ورقمي ٠٧٩١٢٣٤٥٦٧")
        out = os.path.join(self.tmp, "m")
        r = self.cli(g, out, self.store())
        self.assertEqual(r.returncode, 1)
        self.assertIn("needs human review", r.stderr)
        for token in ("سارة", "0791234567", "٠٧٩١٢٣٤٥٦٧"):
            self.assertNotIn(token, r.stderr + r.stdout)
        self.assertFalse(os.path.exists(os.path.join(out, "train.jsonl")))

    def test_withdrawal_still_refused(self):
        g = own_tree(os.path.join(self.tmp, "own"))
        r = self.cli(g, os.path.join(self.tmp, "m"), self.store(withdrawn={1}))
        self.assertEqual(r.returncode, 1)
        self.assertIn("consent withdrawn", r.stderr)
        rows = BM.build_generic(g, "own_recordings", "ar")
        store = CO.load_store(self.store(withdrawn={1}))
        split, _ = BM.assign_splits(rows, 5, fractions=(0.5, 0.25, 0.25))
        self.assertEqual(BM.participant_timing_checks(split), [])   # timing check passes; consent check still refuses
        self.assertTrue(any("withdrawn" in e for e in BM.licence_check(split, "research", store)))


if __name__ == "__main__":
    unittest.main(verbosity=1)
