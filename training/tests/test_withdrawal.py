"""Adversarial synthetic tests for C11 (docs/WITHDRAWAL_DESIGN.md): withdrawal without the token.

Synthetic participants, throw-away peppers, generated tone bytes; no Telegram, network or audio.
  python3 training/tests/test_withdrawal.py
"""
import datetime
import os
import shutil
import sys
import tempfile
import unittest

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from datasets import collection_core as CC  # noqa: E402
from datasets import consent as CO  # noqa: E402
from datasets import pepper as PP  # noqa: E402
from datasets import withdrawal as WD  # noqa: E402

T0 = datetime.datetime(2026, 10, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)
VICTIM, ATTACKER, OTHER = 100000123, 100000456, 100000789


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = Clock()
        self.lineage = CC.LineageIndex()
        self.core = CC.CollectionCore(lambda ref: b"OggS-synthetic-" + ref.encode(), CC.DirStore(os.path.join(self.tmp, "s")),
                                      self.lineage, "consent-ar-en-v0.1", "0" * 64, "campaign-synthetic-1", now=self.clock)
        self.link = PP.LinkTable(os.path.join(self.tmp, "link.sqlite"), PP.InMemoryKms({1: os.urandom(32)}, 1))
        self.svc = WD.WithdrawalService(self.core, self.link, now=self.clock)
        self.cid, self.token = self.enrol(VICTIM, "v")
        self.cid_att, self.token_att = self.enrol(ATTACKER, "a")

    def tearDown(self):
        self.link.close()
        shutil.rmtree(self.tmp)

    def enrol(self, user_id, tag):
        key = self.link.provider.mac(1, PP.message_for(user_id)).hex()
        self.core.start(key)
        self.core.confirm_adult(key)
        for s in CC.SCOPES:
            self.core.answer_scope(key, s, s != "audio_redistribution")
        cid, token = self.core.confirm(key)
        self.link.put(user_id, cid)
        self.core.open_prompt(key, f"p-{tag}")
        decision, path = self.core.media(key, "voice", f"file-{tag}", prompt_id=f"p-{tag}")
        self.assertEqual(decision, "STORED")
        feat = os.path.join(self.tmp, f"features-{tag}.npy")
        with open(feat, "wb") as f:
            f.write(b"synthetic-derived")
        self.lineage.record(self.core.records[cid]["speaker_pseudonym"], feat, "features")
        return cid, token

    def intact(self, cid):
        return self.core.records[cid]["withdrawal"]["status"] == "active" and \
            len(self.lineage.items(self.core.records[cid]["speaker_pseudonym"])) == 2


class IdentifiersAloneNeverDelete(Base):
    def test_attacker_account_naming_victim_ids_in_text(self):
        text = f"delete user {VICTIM} @victim_name consent {self.cid} phone +962791234567"
        r = self.svc.request_from_account(ATTACKER, message_text=text)
        self.assertEqual(r["status"], "CONFIRM_REQUIRED")
        self.svc.confirm_from_account(ATTACKER, r["challenge"])
        self.assertEqual(self.core.records[self.cid_att]["withdrawal"]["status"], "withdrawn")  # attacker's OWN record
        self.assertTrue(self.intact(self.cid))                       # the named victim is untouched

    def test_unknown_account_cannot_reach_any_record(self):
        r = self.svc.request_from_account(OTHER, message_text=f"withdraw {self.cid} for {VICTIM}")
        self.assertEqual(r["status"], "NOT_LINKED")
        self.assertTrue(self.intact(self.cid))

    def test_consent_id_alone_only_quarantines_never_deletes(self):
        r = self.svc.request_unverified(self.cid, "ticket-1")
        self.assertEqual(r["status"], "RECEIVED")
        self.assertTrue(self.intact(self.cid))                       # nothing deleted
        self.assertIn(self.cid, self.svc.quarantined)

    def test_wrong_token_deletes_nothing(self):
        for bad in ("", "x" * 32, self.cid, str(VICTIM), self.token[:-1]):
            with self.assertRaises(WD.WithdrawalError):
                self.svc.by_token(bad)
        self.assertTrue(self.intact(self.cid))


class AccountChallenge(Base):
    def test_valid_flow(self):
        r = self.svc.request_from_account(VICTIM)
        out = self.svc.confirm_from_account(VICTIM, r["challenge"])
        self.assertEqual((out["status"], out["level"]), ("WITHDRAWN", "L2"))
        self.assertEqual({k for k, _ in out["deletion"]["deleted"]}, {"raw_audio", "features"})
        self.assertEqual(out["deletion"]["not_deletable"], list(CC.NOT_DELETABLE))

    def test_challenge_from_other_account_rejected(self):
        r = self.svc.request_from_account(VICTIM)
        with self.assertRaises(WD.WithdrawalError):
            self.svc.confirm_from_account(ATTACKER, r["challenge"])     # stolen challenge string
        self.assertTrue(self.intact(self.cid))

    def test_expired_reused_or_forged_challenge_rejected(self):
        r = self.svc.request_from_account(VICTIM)
        self.clock.t = T0 + WD.CHALLENGE_TTL + datetime.timedelta(seconds=1)
        with self.assertRaises(WD.WithdrawalError):
            self.svc.confirm_from_account(VICTIM, r["challenge"])
        self.assertTrue(self.intact(self.cid))
        r = self.svc.request_from_account(VICTIM)
        self.svc.confirm_from_account(VICTIM, r["challenge"])
        with self.assertRaises(WD.WithdrawalError):
            self.svc.confirm_from_account(VICTIM, r["challenge"])       # single use
        with self.assertRaises(WD.WithdrawalError):
            self.svc.confirm_from_account(VICTIM, "forged-challenge")

    def test_link_lost_falls_back_to_controller_path(self):
        self.link.drop_unmigrated(1)                                    # campaign closed / pepper retired
        self.assertEqual(self.svc.request_from_account(VICTIM)["status"], "NOT_LINKED")
        self.assertTrue(self.intact(self.cid))


class ControllerVerified(Base):
    def open_case(self):
        self.svc.request_unverified(self.cid, "ticket-1")
        return next(iter(self.svc.cases))

    def test_two_distinct_approvals_required(self):
        case = self.open_case()
        self.assertEqual(self.svc.controller_decide(case, "staff-a", True)["status"], "PENDING_SECOND_APPROVAL")
        self.assertTrue(self.intact(self.cid))
        with self.assertRaises(WD.WithdrawalError):
            self.svc.controller_decide(case, "staff-a", True)           # same person twice
        out = self.svc.controller_decide(case, "staff-b", True)
        self.assertEqual((out["status"], out["level"]), ("WITHDRAWN", "L3"))
        self.assertNotIn(self.cid, self.svc.quarantined)
        with self.assertRaises(WD.WithdrawalError):
            self.svc.controller_decide(case, "staff-c", True)           # case closed

    def test_rejection_releases_quarantine_and_keeps_data(self):
        case = self.open_case()
        self.assertEqual(self.svc.controller_decide(case, "staff-a", False)["status"], "RELEASED")
        self.assertNotIn(self.cid, self.svc.quarantined)
        self.assertTrue(self.intact(self.cid))

    def test_no_existence_oracle(self):
        a = self.svc.request_unverified(self.cid, "ticket-1")
        b = self.svc.request_unverified("CNS-ffffffffffff", "ticket-2")
        self.assertEqual(a, b)

    def test_rate_limits(self):
        for i in range(WD.UNVERIFIED_LIMIT_PER_DAY):
            self.svc.request_unverified(f"CNS-{i:012x}", "ticket-flood")
        self.svc.request_unverified(self.cid, "ticket-flood")            # over the requester limit
        self.assertNotIn(self.cid, self.svc.quarantined)
        self.clock.t = T0 + datetime.timedelta(days=1)
        self.svc.request_unverified(self.cid, "ticket-flood")
        self.assertIn(self.cid, self.svc.quarantined)

    def test_quarantine_blocks_new_manifests(self):
        self.svc.request_unverified(self.cid, "ticket-1")
        rec = self.core.records[self.cid]
        row = {"corpus": "own_recordings", "speaker": rec["speaker_pseudonym"], "consent_id": self.cid, "split": "train"}
        store = {self.cid: rec}
        self.assertEqual(CO.check_rows([row], store, "research", today="2026-10-12"), [])
        self.assertTrue(CO.check_rows([row], store, "research", today="2026-10-12", quarantined=self.svc.quarantined))


class AuditAndIdempotence(Base):
    def test_audit_has_no_telegram_ids_tokens_or_macs(self):
        r = self.svc.request_from_account(VICTIM, message_text=f"{ATTACKER}")
        self.svc.confirm_from_account(VICTIM, r["challenge"])
        self.svc.request_unverified(self.cid_att, "ticket-9")
        with self.assertRaises(WD.WithdrawalError):
            self.svc.by_token("wrong")
        text = repr(self.svc.audit)
        mac = self.link.provider.mac(1, PP.message_for(VICTIM)).hex()
        for secret in (str(VICTIM), str(ATTACKER), self.token, self.token_att, mac, r["challenge"]):
            self.assertNotIn(secret, text)
        self.assertIn("withdrawn", text)

    def test_repeat_withdrawal_is_idempotent(self):
        self.svc.by_token(self.token)
        self.assertEqual(self.svc.by_token(self.token)["status"], "ALREADY_WITHDRAWN")
        self.assertEqual(self.svc.request_from_account(VICTIM)["status"], "ALREADY_WITHDRAWN")


if __name__ == "__main__":
    unittest.main(verbosity=1)
