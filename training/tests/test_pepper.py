"""Adversarial synthetic tests for C9 (docs/PEPPER_DESIGN.md): pepper kept out of the DB, logs,
manifests and error messages; DB copy alone cannot re-link; missing/invalid secret fails
closed; rotation and loss behave as documented.

Throw-away peppers from os.urandom in temp dirs; synthetic user ids; no network, no audio.
  python3 training/tests/test_pepper.py
"""
import base64
import datetime
import hashlib
import io
import logging
import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
import unittest

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from datasets import collection_core as CC  # noqa: E402
from datasets import pepper as PP  # noqa: E402

USERS = [100000001 + 7919 * i for i in range(25)]          # synthetic "Telegram ids"
NOW = datetime.datetime(2026, 10, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)


def forms(secret):
    """Every textual form of a secret an attacker might grep for."""
    return [secret, secret.hex().encode(), base64.b64encode(secret), base64.urlsafe_b64encode(secret),
            secret.hex().upper().encode()]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.tmp, "db")
        self.secret_dir = os.path.join(self.tmp, "secrets")
        os.makedirs(self.db_dir)
        os.makedirs(self.secret_dir, mode=0o700)
        self.db = os.path.join(self.db_dir, "link.sqlite")
        self.p1 = os.urandom(32)
        self.f1 = self.write_secret("v1", self.p1)
        self.log_buf = io.StringIO()
        self.handler = logging.StreamHandler(self.log_buf)
        logging.getLogger("voise").addHandler(self.handler)
        logging.getLogger("voise").setLevel(logging.DEBUG)

    def tearDown(self):
        logging.getLogger("voise").removeHandler(self.handler)
        shutil.rmtree(self.tmp)

    def write_secret(self, name, value, mode=0o600, where=None):
        p = os.path.join(where or self.secret_dir, name)
        with open(p, "wb") as f:
            f.write(value)
        os.chmod(p, mode)
        return p

    def provider(self, files=None, current=1):
        return PP.DevFilePepper(files or {1: self.f1}, current, self.db)

    def macs(self, secret):
        return [PP.hmac.new(secret, PP.message_for(u), hashlib.sha256).hexdigest() for u in USERS]


class SecretNeverPersistedOrLogged(Base):
    def test_db_file_holds_no_pepper_and_no_user_ids(self):
        lt = PP.LinkTable(self.db, self.provider())
        for i, u in enumerate(USERS):
            lt.put(u, f"CNS-{i:012x}")
        lt.close()
        with open(self.db, "rb") as f:
            blob = f.read()
        for form in forms(self.p1):
            self.assertNotIn(form, blob)
        for u in USERS:
            self.assertNotIn(str(u).encode(), blob)

    def test_logs_hold_no_pepper_ids_or_macs(self):
        lt = PP.LinkTable(self.db, self.provider())
        for i, u in enumerate(USERS[:5]):
            lt.put(u, f"CNS-{i:012x}")
            lt.lookup(u)
        lt.lookup(999999999)
        text = self.log_buf.getvalue()
        self.assertIn("link.put", text)                       # logging is active
        for form in forms(self.p1):
            self.assertNotIn(form.decode("latin-1"), text)
        for u in USERS[:5]:
            self.assertNotIn(str(u), text)
        for m in self.macs(self.p1)[:5]:
            self.assertNotIn(m, text)

    def test_manifest_rows_and_consent_records_hold_no_pepper_or_link_macs(self):
        lt = PP.LinkTable(self.db, self.provider())
        core = CC.CollectionCore(lambda ref: b"OggS-synthetic", CC.DirStore(os.path.join(self.tmp, "store")),
                                 CC.LineageIndex(), "consent-ar-en-v0.1", "0" * 64, "campaign-synthetic-1",
                                 now=lambda: NOW)
        rows = []
        for u in USERS[:3]:
            key = lt.provider.mac(1, PP.message_for(u)).hex()
            core.start(key)
            core.confirm_adult(key)
            for s in CC.SCOPES:
                core.answer_scope(key, s, s != "audio_redistribution")
            cid, _ = core.confirm(key)
            lt.put(u, cid)
            rec = core.records[cid]
            rows.append({"corpus": "own_recordings", "speaker": rec["speaker_pseudonym"], "consent_id": cid,
                         "split": "train", "path": "x.wav", "session": "ses-00000001"})
        text = repr(rows) + repr(core.records)
        for form in forms(self.p1):
            self.assertNotIn(form.decode("latin-1"), text)
        for m in self.macs(self.p1)[:3]:
            self.assertNotIn(m, text)
            self.assertNotIn(m[:10], text)                    # pseudonyms are random, not MAC-derived
        for u in USERS[:3]:
            self.assertNotIn(str(u), text)

    def test_repr_str_pickle_do_not_expose_secret(self):
        p = self.provider()
        for s in (repr(p), str(p), repr(PP.LinkTable(self.db, p).provider)):
            for form in forms(self.p1):
                self.assertNotIn(form.decode("latin-1"), s)
        with self.assertRaises(TypeError):
            pickle.dumps(p)


class FailClosed(Base):
    def test_missing_secret_file(self):
        with self.assertRaises(PP.PepperUnavailable) as cm:
            PP.DevFilePepper({1: os.path.join(self.secret_dir, "absent")}, 1, self.db)
        self.assertNotIn(self.secret_dir, str(cm.exception))

    def test_link_table_requires_provider(self):
        for bad in (None, b"raw-bytes", {"1": "x"}):
            with self.assertRaises(PP.PepperUnavailable):
                PP.LinkTable(self.db, bad)

    def test_invalid_pepper_values_rejected_without_echo(self):
        for name, value in (("short", os.urandom(16)), ("zeros", b"\0" * 64), ("pattern", b"ab" * 32)):
            f = self.write_secret(name, value)
            with self.assertRaises(PP.PepperPolicyError) as cm:
                PP.DevFilePepper({1: f}, 1, self.db)
            for form in forms(value):
                self.assertNotIn(form.decode("latin-1"), str(cm.exception))
        with self.assertRaises(PP.PepperPolicyError):
            PP.validate_pepper("a" * 64)                       # str is not bytes

    def test_bad_permissions_location_or_type_rejected(self):
        loose = self.write_secret("loose", os.urandom(32), mode=0o644)
        with self.assertRaises(PP.PepperPolicyError):
            PP.DevFilePepper({1: loose}, 1, self.db)
        in_db = self.write_secret("pepper", os.urandom(32), where=self.db_dir)
        with self.assertRaises(PP.PepperPolicyError):
            PP.DevFilePepper({1: in_db}, 1, self.db)
        link = os.path.join(self.secret_dir, "link_to_db_pepper")
        os.symlink(in_db, link)
        with self.assertRaises(PP.PepperPolicyError):
            PP.DevFilePepper({1: link}, 1, self.db)            # symlink resolved to the DB dir
        with self.assertRaises(PP.PepperPolicyError):
            PP.DevFilePepper({1: self.secret_dir}, 1, self.db) # a directory, not a file
        repo_like = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(repo_like, ".git"))
        in_repo = self.write_secret("pepper", os.urandom(32), where=repo_like)
        with self.assertRaises(PP.PepperPolicyError):
            PP.DevFilePepper({1: in_repo}, 1, self.db)

    def test_invalid_user_ids_rejected_without_echo(self):
        lt = PP.LinkTable(self.db, self.provider())
        for bad in (True, -5, 0, "123456789", 12.5, None):
            with self.assertRaises(ValueError) as cm:
                lt.put(bad, "CNS-000000000000")
            self.assertNotIn(str(bad), str(cm.exception).replace("positive integer", ""))

    def test_unknown_version_error_has_no_secret(self):
        p = self.provider()
        with self.assertRaises(PP.PepperUnavailable) as cm:
            p.mac(7, PP.message_for(USERS[0]))
        for form in forms(self.p1):
            self.assertNotIn(form.decode("latin-1"), str(cm.exception) + repr(cm.exception))


class DatabaseAloneCannotRelink(Base):
    def setUp(self):
        super().setUp()
        lt = PP.LinkTable(self.db, self.provider())
        for i, u in enumerate(USERS):
            lt.put(u, f"CNS-{i:012x}")
        lt.close()
        self.copy = os.path.join(self.tmp, "stolen_backup.sqlite")
        shutil.copy(self.db, self.copy)                       # attacker has the DB (credentials or backup)
        self.space = range(100000000, 100000001 + 7919 * 30)  # the synthetic id space contains every user

    def stolen_macs(self):
        con = sqlite3.connect(self.copy)
        out = {r[0] for r in con.execute("SELECT mac FROM link")}
        con.close()
        return out

    def brute(self, keyfn):
        stolen = self.stolen_macs()
        return sum(1 for u in self.space[::7919] if keyfn(PP.message_for(u)) in stolen) + \
            sum(1 for u in USERS if keyfn(PP.message_for(u)) in stolen)

    def test_db_copy_without_pepper_matches_nothing(self):
        self.assertEqual(self.brute(lambda m: hashlib.sha256(m).hexdigest()), 0)            # unkeyed hash
        guess = os.urandom(32)
        self.assertEqual(self.brute(lambda m: PP.hmac.new(guess, m, hashlib.sha256).hexdigest()), 0)

    def test_db_role_cannot_build_a_lookup(self):
        with self.assertRaises(PP.PepperUnavailable):
            PP.LinkTable(self.copy, None)
        con = sqlite3.connect(self.copy)
        cols = [r[1] for r in con.execute("PRAGMA table_info(link)")]
        con.close()
        self.assertEqual(cols, ["mac", "pepper_version", "consent_id"])  # no id, name, pepper column

    def test_residual_risk_with_pepper_relinking_is_possible(self):
        # Pseudonymisation, not anonymity: whoever holds the pepper re-links every row.
        self.assertEqual(self.brute(lambda m: PP.hmac.new(self.p1, m, hashlib.sha256).hexdigest()), len(USERS))


class RotationAndLoss(Base):
    def test_rotation_migrates_on_contact_and_guards_retirement(self):
        lt = PP.LinkTable(self.db, self.provider())
        lt.put(USERS[0], "CNS-00000000000a")
        lt.put(USERS[1], "CNS-00000000000b")
        lt.close()
        p2 = os.urandom(32)
        f2 = self.write_secret("v2", p2)
        lt = PP.LinkTable(self.db, self.provider({1: self.f1, 2: f2}, current=2))
        self.assertEqual(lt.lookup(USERS[0]), "CNS-00000000000a")            # found via v1, migrated to v2
        self.assertEqual(lt.rows_on_version(2), 1)
        self.assertEqual(lt.rows_on_version(1), 1)                           # USERS[1] has not returned yet
        self.assertFalse(lt.can_retire(1))
        lt.put(USERS[2], "CNS-00000000000c")
        self.assertEqual(lt.rows_on_version(2), 2)                           # new rows use the current version
        self.assertEqual(lt.drop_unmigrated(1), 1)                           # campaign close / forced retirement
        self.assertTrue(lt.can_retire(1))
        lt.close()
        lt = PP.LinkTable(self.db, self.provider({2: f2}, current=2))        # v1 destroyed
        self.assertEqual(lt.lookup(USERS[0]), "CNS-00000000000a")
        self.assertIsNone(lt.lookup(USERS[1]))                               # documented: falls back to token (C11)

    def test_loss_of_current_pepper_fails_closed(self):
        with self.assertRaises(PP.PepperUnavailable):
            PP.DevFilePepper({1: self.f1}, 2, self.db)                       # current version missing
        lt = PP.LinkTable(self.db, self.provider())
        lt.put(USERS[0], "CNS-00000000000a")
        lt.close()
        other = self.write_secret("other", os.urandom(32))
        lt = PP.LinkTable(self.db, self.provider({1: other}))                # wrong / replaced secret
        self.assertIsNone(lt.lookup(USERS[0]))                               # no silent weaker fallback
        with sqlite3.connect(self.db) as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM link").fetchone()[0], 1)

    def test_kms_style_provider_keeps_key_inside(self):
        kms = PP.InMemoryKms({1: os.urandom(32)}, 1)
        lt = PP.LinkTable(self.db, kms)
        lt.put(USERS[3], "CNS-00000000000d")
        self.assertEqual(lt.lookup(USERS[3]), "CNS-00000000000d")
        self.assertNotIn("key", repr(kms).lower().replace("kms", ""))


if __name__ == "__main__":
    unittest.main(verbosity=1)
