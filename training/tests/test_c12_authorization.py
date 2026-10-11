"""C12 adversarial tests: authorization matrix + durable withdrawal/collection state.

Synthetic principals, throw-away peppers, synthetic bytes; no Telegram, network or real audio.
  python3 training/tests/test_c12_authorization.py
"""
import datetime
import hashlib
import itertools
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from datasets import authz as AZ  # noqa: E402
from datasets import consent as CO  # noqa: E402
from datasets import pepper as PP  # noqa: E402
from datasets import secure_state as SS  # noqa: E402

T0 = datetime.datetime(2026, 10, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)
VICTIM, ATTACKER, STRANGER = 100000123, 100000456, 100000789


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class Signer:
    def sign(self, body):
        return b"sig:" + hashlib.sha256(repr(body).encode()).digest()


class World:
    """One deployment's worth of reference components on local files (re-creatable = 'restart')."""

    def __init__(self, tmp, clock, kms=None):
        self.tmp, self.clock = tmp, clock
        self.kms = kms or PP.InMemoryKms({1: os.urandom(32)}, 1)
        self.reg = AZ.Registry(os.path.join(tmp, "registry.sqlite"))
        self.state = SS.StateDB(os.path.join(tmp, "state.sqlite"), now=clock)
        self.store = SS.ObjectStore(self.reg, os.path.join(tmp, "objects"), self.state)
        self.gw = SS.KmsGateway(self.reg, self.kms, decrypt_fn=lambda ct: b"plain:" + ct,
                                collector_signer=Signer(), processing_signer=Signer())

    def bind(self, P):
        self.P = P
        self.link_col = PP.LinkTable(os.path.join(self.tmp, "link.sqlite"), SS.GatewayMacProvider(self.gw, P["col"]))
        self.link_wd = PP.LinkTable(os.path.join(self.tmp, "link.sqlite"), SS.GatewayMacProvider(self.gw, P["wd"]))
        self.writer = SS.CollectionWriter(self.reg, self.state, self.store, P["col"])
        self.wd = SS.WithdrawalService(self.reg, self.state, self.link_wd, self.gw, P["wd"])
        self.review = SS.CaseReview(self.reg, self.state)
        self.exec = SS.DeletionExecutor(self.reg, self.state, self.store)
        return self

    def close(self):
        self.link_col.close()
        self.link_wd.close()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = Clock()
        self.w = World(self.tmp, self.clock)
        reg = self.w.reg
        self.P = {"col": reg.provision("svc-collection", AZ.COLLECTION, "svc-col"),
                  "proc": reg.provision("svc-processing", AZ.PROCESSING, "svc-proc"),
                  "wd": reg.provision("svc-withdrawal", AZ.WITHDRAWAL, "svc-wd"),
                  "alice": reg.provision("staff-alice", AZ.REVIEWER, "person-alice"),
                  "bob": reg.provision("staff-bob", AZ.APPROVER, "person-bob"),
                  "alice2": reg.provision("staff-alice-2", AZ.APPROVER, "person-alice"),   # same human, 2nd account
                  "audit": reg.provision("staff-audit", AZ.AUDITOR, "person-carol"),
                  "kms": reg.provision("svc-kms", AZ.KMS, "svc-kms")}
        self.w.bind(self.P)
        self.secrets_seen = []
        self.v = self.enrol(VICTIM, "a")
        self.a = self.enrol(ATTACKER, "b")

    def tearDown(self):
        self.w.close()
        shutil.rmtree(self.tmp)

    def enrol(self, user_id, tag):
        cid, pseud, token = f"CNS-{tag * 12}", f"own_recordings:spk-{tag * 10}", f"tok-{os.urandom(12).hex()}"
        self.w.writer.register_consent(cid, pseud, token)
        self.w.link_col.put(user_id, cid)
        self.w.writer.store_audio(pseud, "p1.ogg", b"OggS-synthetic-" + tag.encode())
        derived = self.w.store.put_derived(self.P["proc"], pseud, "derived-features.npy", b"synthetic-features")
        SS.register_derived(self.w.reg, self.w.state, self.P["proc"], pseud, "features", derived)
        self.secrets_seen += [token, str(user_id)]
        return {"cid": cid, "pseud": pseud, "token": token, "user": user_id}

    def status(self, cid):
        with self.w.state.tx() as c:
            return c.execute("SELECT status FROM consents WHERE consent_id=?", (cid,)).fetchone()[0]

    def objects(self, pseud):
        return self.w.store.list(self.P["proc"], pseud)

    def open_case(self, who=None):
        who = who or self.v
        self.w.wd.request_unverified(who["cid"], "ticket-1")
        with self.w.state.tx() as c:
            return c.execute("SELECT case_id FROM cases WHERE consent_id=? AND state='open'", (who["cid"],)).fetchone()[0]


# 1 --------------------------------------------------------------- every forbidden role/operation pair
class Matrix(Base):
    def test_every_role_operation_pair(self):
        role_of = {AZ.COLLECTION: "col", AZ.PROCESSING: "proc", AZ.WITHDRAWAL: "wd", AZ.REVIEWER: "alice",
                   AZ.APPROVER: "bob", AZ.AUDITOR: "audit", AZ.KMS: "kms"}
        checked = 0
        for role, op in itertools.product(AZ.ROLES, AZ.OPERATIONS):
            with self.subTest(role=role, op=op):
                if op in AZ.MATRIX[role]:
                    self.w.reg.authorize(self.P[role_of[role]], op)
                else:
                    with self.assertRaises(AZ.Forbidden):
                        self.w.reg.authorize(self.P[role_of[role]], op)
                checked += 1
        self.assertEqual(checked, len(AZ.ROLES) * len(AZ.OPERATIONS))
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.authorize(self.P["proc"], "no.such.operation")       # deny by default

    def test_service_boundaries_enforce_the_matrix(self):
        v = self.v
        calls = [  # (callable(principal), allowed principals)
            (lambda p: self.w.store.get(p, v["pseud"], "p1.ogg"), {"proc"}),
            (lambda p: self.w.store.list(p, v["pseud"]), {"proc"}),
            (lambda p: self.w.store.put(p, v["pseud"], f"x-{p.principal_id}.ogg", b"x"), {"col"}),
            (lambda p: self.w.store.put_derived(p, v["pseud"], f"derived-{p.principal_id}", b"x"), {"proc"}),
            (lambda p: self.w.gw.mac(p, 1, PP.message_for(STRANGER)), {"col", "wd"}),
            (lambda p: self.w.gw.decrypt(p, b"ct"), {"proc"}),
            (lambda p: self.w.gw.sign_store_entry(p, {"a": 1}), {"col"}),
            (lambda p: self.w.gw.sign_processing_entry(p, {"a": 1}), {"proc"}),
            (lambda p: SS.read_audit(self.w.reg, self.w.state, p), {"audit"}),
            (lambda p: SS.deletion_status(self.w.reg, self.w.state, p, v["cid"]), {"proc", "wd", "audit"}),
            (lambda p: self.w.state.quarantined_ids(self.w.reg, p), {"proc"}),
            (lambda p: SS.register_derived(self.w.reg, self.w.state, p, v["pseud"], "f", f"/x/{p.principal_id}"), {"col", "proc"}),
            (lambda p: self.w.review.read(p, "CASE-none"), {"alice", "bob", "alice2", "audit"}),
        ]
        for fn, allowed in calls:
            for name, p in self.P.items():
                with self.subTest(fn=fn, principal=name):
                    if name in allowed:
                        fn(p)
                    else:
                        with self.assertRaises(AZ.Forbidden):
                            fn(p)

    def test_no_role_can_export_keys(self):
        for name, p in self.P.items():
            with self.subTest(principal=name), self.assertRaises(AZ.Forbidden):
                self.w.gw.export_key(p, "pepper")


# 2 --------------------------------------------------------------- privilege escalation attempts
class Escalation(Base):
    def test_forged_or_mixed_credentials(self):
        forged = AZ.Principal("staff-bob", self.P["alice"]._credential)       # bob's id, alice's secret
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.authorize(forged, "case.read")
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.authorize(AZ.Principal("svc-processing", "guess"), "object.get")

    def test_role_cannot_be_supplied_by_caller(self):
        with self.assertRaises(AttributeError):
            self.P["alice"].role = AZ.APPROVER                                 # handle carries no role
        class Duck:
            principal_id, _credential, role = "staff-alice", None, AZ.PROCESSING
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.authorize(Duck(), "object.get")
        with self.assertRaises(AZ.Forbidden):
            self.w.review.approve(self.P["alice"], self.open_case())           # first ok …
            self.w.review.approve(self.P["alice"], self.open_case())

    def test_deactivated_principal(self):
        self.w.reg.deactivate("svc-processing")
        with self.assertRaises(AZ.Forbidden):
            self.w.store.list(self.P["proc"], self.v["pseud"])

    def test_participant_ids_in_text_are_ignored(self):
        r = self.w.wd.request_from_account(ATTACKER, message_text=f"withdraw {VICTIM} {self.v['cid']}")
        self.w.wd.confirm_from_account(ATTACKER, r["challenge"])
        self.assertEqual(self.status(self.a["cid"]), "withdrawn")
        self.assertEqual(self.status(self.v["cid"]), "active")


# 3 --------------------------------------------------------------- cross-participant access / deletion
class CrossParticipant(Base):
    def test_challenge_stolen_by_other_account(self):
        r = self.w.wd.request_from_account(VICTIM)
        with self.assertRaises(SS.WithdrawalError):
            self.w.wd.confirm_from_account(ATTACKER, r["challenge"])
        self.assertEqual(self.status(self.v["cid"]), "active")

    def test_deleting_another_pseudonyms_object_under_own_job(self):
        self.w.wd.by_token(self.a["token"])                                    # attacker's own job exists
        victim_obj = self.objects(self.v["pseud"])[0]
        with self.assertRaises(AZ.Forbidden):
            self.w.store.delete(self.P["proc"], self.a["pseud"], victim_obj)
        with self.assertRaises(AZ.Forbidden):
            self.w.store.delete(self.P["proc"], self.v["pseud"], victim_obj)   # victim has no job
        self.assertEqual(len(self.objects(self.v["pseud"])), 2)

    def test_executor_refuses_without_job(self):
        with self.assertRaises(AZ.Forbidden):
            self.w.exec.run(self.P["proc"], self.v["cid"])
        self.assertEqual(len(self.objects(self.v["pseud"])), 2)

    def test_path_traversal_refused(self):
        with self.assertRaises(AZ.Forbidden):
            self.w.store.get(self.P["proc"], self.v["pseud"], "../" + self.a["pseud"].replace(":", "_") + "/p1.ogg")
        with self.assertRaises(AZ.Forbidden):
            self.w.store.get(self.P["proc"], "own_recordings:spk-../../etc", "passwd")
        alias = "own_recordings:spk-../" + self.v["pseud"].replace(":", "_")    # alias into victim's folder
        with self.assertRaises(AZ.Forbidden):
            self.w.store.put(self.P["col"], alias, "planted.ogg", b"x")
        with self.assertRaises(AZ.Forbidden):
            self.w.store.list(self.P["proc"], alias)
        for bad in ("/abs.ogg", ".hidden", "a/b.ogg", "a..b", ""):
            with self.subTest(name=bad), self.assertRaises(AZ.Forbidden):
                self.w.store.put(self.P["col"], self.v["pseud"], bad, b"x")


# 4 --------------------------------------------------------------- one person cannot satisfy both L3 approvals
class Independence(Base):
    def test_single_reviewer_both_approvals(self):
        case = self.open_case()
        self.assertEqual(self.w.review.approve(self.P["alice"], case)["status"], "PENDING_SECOND_APPROVAL")
        with self.assertRaises(AZ.Forbidden):
            self.w.review.approve(self.P["alice"], case)                       # same account again
        with self.assertRaises(AZ.Forbidden):
            self.w.review.approve(self.P["alice2"], case)                      # same person, approver account
        self.assertEqual(self.status(self.v["cid"]), "active")
        out = self.w.review.approve(self.P["bob"], case)
        self.assertEqual(out["status"], "WITHDRAWN_DELETION_PENDING")

    def test_second_before_first_and_wrong_roles(self):
        case = self.open_case()
        with self.assertRaises(SS.WithdrawalError):
            self.w.review.approve(self.P["bob"], case)                         # no first approval yet
        for name in ("col", "proc", "wd", "audit", "kms"):
            with self.subTest(name=name), self.assertRaises(AZ.Forbidden):
                self.w.review.approve(self.P[name], case)
        with self.assertRaises(AZ.Forbidden):
            self.w.review.reject(self.P["bob"], case)                          # approver cannot reject

    def test_approvals_are_immutable(self):
        case = self.open_case()
        self.w.review.approve(self.P["alice"], case)
        con = sqlite3.connect(self.w.state.path)
        with self.assertRaises(sqlite3.DatabaseError):
            con.execute("UPDATE approvals SET person_ref='person-zed'")
        with self.assertRaises(sqlite3.DatabaseError):
            con.execute("DELETE FROM approvals")
        con.close()


# 5/6 ------------------------------------------------------------- concurrency, replay, restart
class ChallengesConcurrencyRestart(Base):
    def test_concurrent_consumption_exactly_once(self):
        r = self.w.wd.request_from_account(VICTIM)
        results, lock = [], threading.Lock()

        def worker():
            w = World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)      # separate instance + connections
            try:
                out = w.wd.confirm_from_account(VICTIM, r["challenge"])["status"]
            except SS.WithdrawalError:
                out = "REJECTED"
            finally:
                w.close()
            with lock:
                results.append(out)
        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count("WITHDRAWN_DELETION_PENDING"), 1, results)
        self.assertEqual(results.count("REJECTED"), 11)
        with self.w.state.tx() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM deletion_jobs").fetchone()[0], 1)

    def test_restart_expired_forged_consumed(self):
        r1 = self.w.wd.request_from_account(VICTIM)
        r2 = self.w.wd.request_from_account(VICTIM)
        self.w.wd.confirm_from_account(VICTIM, r2["challenge"])                # consumed (and withdrawn)
        self.w.close()
        self.w = World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)     # restart
        with self.assertRaises(SS.WithdrawalError):
            self.w.wd.confirm_from_account(VICTIM, r2["challenge"])            # replay after restart
        with self.assertRaises(SS.WithdrawalError):
            self.w.wd.confirm_from_account(VICTIM, "forged-" + r1["challenge"])
        r3 = self.w.wd.request_from_account(ATTACKER)
        self.clock.t = T0 + SS.CHALLENGE_TTL + datetime.timedelta(seconds=1)
        self.w.close()
        self.w = World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)
        with self.assertRaises(SS.WithdrawalError):
            self.w.wd.confirm_from_account(ATTACKER, r3["challenge"])          # expired across restart
        self.assertEqual(self.status(self.a["cid"]), "active")

    def test_db_holds_only_challenge_hashes(self):
        r = self.w.wd.request_from_account(VICTIM)
        with open(self.w.state.path, "rb") as f:
            blob = f.read()
        self.assertNotIn(r["challenge"].encode(), blob)
        self.assertIn(hashlib.sha256(r["challenge"].encode()).hexdigest().encode(), blob)


# 7 --------------------------------------------------------------- quarantine survives restart
class QuarantinePersistence(Base):
    def test_quarantine_survives_restart_and_blocks_manifests(self):
        self.open_case()
        self.w.close()
        self.w = World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)
        q = self.w.state.quarantined_ids(self.w.reg, self.P["proc"])
        self.assertEqual(q, {self.v["cid"]})
        rec = {"consent_id": self.v["cid"]}
        row = {"corpus": "own_recordings", "speaker": self.v["pseud"], "consent_id": self.v["cid"], "split": "train"}
        errs = CO.check_rows([row], {self.v["cid"]: rec}, "research", today="2026-10-12", quarantined=q)
        self.assertTrue(any("quarantined" in e for e in errs))
        self.assertEqual(len(self.objects(self.v["pseud"])), 2)                # quarantine deletes nothing

    def test_no_existence_oracle_and_rate_limit_persist(self):
        a = self.w.wd.request_unverified(self.v["cid"], "t1")
        b = self.w.wd.request_unverified("CNS-ffffffffffff", "t2")
        self.assertEqual(a, b)
        for i in range(SS.UNVERIFIED_LIMIT_PER_DAY):
            self.w.wd.request_unverified(f"CNS-{i:012x}", "flood")
        self.w.close()
        self.w = World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)
        self.w.wd.request_unverified(self.a["cid"], "flood")                  # counter survived restart
        self.assertNotIn(self.a["cid"], self.w.state.quarantined_ids(self.w.reg, self.P["proc"]))


# 8 --------------------------------------------------------------- storage errors fail closed
class FailClosed(Base):
    def test_corrupt_state_db(self):
        self.w.close()
        with open(os.path.join(self.tmp, "state.sqlite"), "wb") as f:
            f.write(b"this is not a database" * 100)
        with self.assertRaises(AZ.StoreUnavailable):
            SS.StateDB(os.path.join(self.tmp, "state.sqlite"), now=self.clock)
        self.w.state = SS.StateDB.__new__(SS.StateDB)
        self.w.state.path, self.w.state.now = os.path.join(self.tmp, "state.sqlite"), self.clock
        wd = SS.WithdrawalService(self.w.reg, self.w.state, self.w.link_wd, self.w.gw, self.P["wd"])
        with self.assertRaises(AZ.StoreUnavailable):
            wd.by_token(self.v["token"])
        self.w.link_col = PP.LinkTable(os.path.join(self.tmp, "link.sqlite"), SS.GatewayMacProvider(self.w.gw, self.P["col"]))
        self.w.link_wd = PP.LinkTable(os.path.join(self.tmp, "link.sqlite"), SS.GatewayMacProvider(self.w.gw, self.P["wd"]))
        self.assertEqual(len(self.objects(self.v["pseud"])), 2)

    def test_corrupt_registry_denies_everything(self):
        with open(self.w.reg.path, "wb") as f:
            f.write(b"garbage" * 200)
        with self.assertRaises((AZ.StoreUnavailable, AZ.Forbidden)):
            self.w.store.list(self.P["proc"], self.v["pseud"])
        with self.assertRaises((AZ.StoreUnavailable, AZ.Forbidden)):
            self.w.wd.by_token(self.v["token"])

    def test_failure_inside_transaction_rolls_back(self):
        orig = self.w.state.audit

        def boom(c, actor, event, **kw):
            if event == "withdrawn":
                raise sqlite3.OperationalError("disk I/O error (injected)")
            return orig(c, actor, event, **kw)
        self.w.state.audit = boom
        with self.assertRaises(AZ.StoreUnavailable):
            self.w.wd.by_token(self.v["token"])
        self.w.state.audit = orig
        self.assertEqual(self.status(self.v["cid"]), "active")                # no partial withdrawal
        with self.w.state.tx() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM deletion_jobs").fetchone()[0], 0)


# 9 --------------------------------------------------------------- audit hygiene
class AuditHygiene(Base):
    def test_audit_and_db_files_hold_no_secrets(self):
        r = self.w.wd.request_from_account(VICTIM)
        self.w.wd.confirm_from_account(VICTIM, r["challenge"])
        case = self.open_case(self.a)
        self.w.review.approve(self.P["alice"], case)
        with self.assertRaises(SS.WithdrawalError):
            self.w.wd.by_token("wrong-token")
        rows = SS.read_audit(self.w.reg, self.w.state, self.P["audit"])
        text = repr(rows)
        macs = [self.w.kms.mac(1, PP.message_for(u)).hex() for u in (VICTIM, ATTACKER)]
        banned = self.secrets_seen + [r["challenge"], self.v["cid"], self.a["cid"]] + macs + [p._credential for p in self.P.values()]
        for b in banned:
            self.assertNotIn(b, text)
        self.assertIn("withdrawn", text)
        for path in (self.w.state.path, self.w.reg.path):
            with open(path, "rb") as f:
                blob = f.read()
            for b in self.secrets_seen + [r["challenge"]] + [p._credential for p in self.P.values()]:
                self.assertNotIn(b.encode(), blob, path)

    def test_audit_is_append_only(self):
        self.w.wd.by_token(self.v["token"])
        con = sqlite3.connect(self.w.state.path)
        with self.assertRaises(sqlite3.DatabaseError):
            con.execute("UPDATE audit SET event='nothing'")
        with self.assertRaises(sqlite3.DatabaseError):
            con.execute("DELETE FROM audit")
        con.close()


# 10 -------------------------------------------------------------- keys / pepper / objects / deletion records
class KeyAndRecordAccess(Base):
    def test_pepper_operations_need_authorized_identity(self):
        lt = PP.LinkTable(os.path.join(self.tmp, "link.sqlite"), SS.GatewayMacProvider(self.w.gw, self.P["alice"]))
        with self.assertRaises(AZ.Forbidden):
            lt.lookup(VICTIM)
        lt.close()
        for name in ("proc", "alice", "bob", "audit", "kms"):
            with self.subTest(name=name), self.assertRaises(AZ.Forbidden):
                self.w.gw.mac(self.P[name], 1, PP.message_for(VICTIM))

    def test_encrypted_objects_and_deletion_records(self):
        for name in ("col", "wd", "alice", "bob", "audit", "kms"):
            with self.subTest(name=name):
                with self.assertRaises(AZ.Forbidden):
                    self.w.store.get(self.P[name], self.v["pseud"], "p1.ogg")
                with self.assertRaises(AZ.Forbidden):
                    self.w.gw.decrypt(self.P[name], b"ct")
                with self.assertRaises(AZ.Forbidden):
                    self.w.exec.run(self.P[name], self.v["cid"])
        for name in ("col", "alice", "bob", "kms"):
            with self.subTest(name=name), self.assertRaises(AZ.Forbidden):
                SS.deletion_status(self.w.reg, self.w.state, self.P[name], self.v["cid"])

    def test_collection_is_write_once(self):
        with self.assertRaises(AZ.Forbidden):
            self.w.store.put(self.P["col"], self.v["pseud"], "p1.ogg", b"overwrite")


# 11/12 ----------------------------------------------------------- incomplete index, crash and retry
class DeletionSafety(Base):
    def test_full_deletion_done(self):
        self.w.wd.by_token(self.v["token"])
        out = self.w.exec.run(self.P["proc"], self.v["cid"])
        self.assertEqual(out["status"], "DONE")
        self.assertEqual(self.objects(self.v["pseud"]), [])
        self.assertEqual(out["not_deletable"], list(SS.NOT_DELETABLE))
        self.assertEqual(len(self.objects(self.a["pseud"])), 2)                # other participant untouched

    def test_incomplete_index_never_reports_success(self):
        self.w.store.put(self.P["col"], self.v["pseud"], "unindexed.ogg", b"pipeline forgot to register me")
        self.w.wd.by_token(self.v["token"])
        out = self.w.exec.run(self.P["proc"], self.v["cid"])
        self.assertEqual((out["status"], out["unindexed_objects"]), ("INCOMPLETE", 1))
        self.assertEqual(SS.deletion_status(self.w.reg, self.w.state, self.P["wd"], self.v["cid"]), "incomplete")
        self.assertEqual([os.path.basename(p) for p in self.objects(self.v["pseud"])], ["unindexed.ogg"])

    def test_crash_mid_deletion_then_retry(self):
        self.w.wd.by_token(self.v["token"])
        orig, calls = self.w.store.delete, {"n": 0}

        def crashing(principal, pseud, path):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("power loss (injected)")
            return orig(principal, pseud, path)
        self.w.store.delete = crashing
        with self.assertRaises(OSError):
            self.w.exec.run(self.P["proc"], self.v["cid"])
        self.assertEqual(SS.deletion_status(self.w.reg, self.w.state, self.P["wd"], self.v["cid"]), "in_progress")
        self.w.store.delete = orig
        self.w.close()
        self.w = World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)     # restart, then retry
        self.assertEqual(self.w.exec.run(self.P["proc"], self.v["cid"])["status"], "DONE")
        self.assertEqual(self.w.exec.run(self.P["proc"], self.v["cid"])["status"], "DONE")   # idempotent
        self.assertEqual(self.objects(self.v["pseud"]), [])

    def test_retry_of_withdrawal_does_not_duplicate_or_reopen(self):
        self.assertEqual(self.w.wd.by_token(self.v["token"])["status"], "WITHDRAWN_DELETION_PENDING")
        self.assertEqual(self.w.wd.by_token(self.v["token"])["status"], "ALREADY_WITHDRAWN")
        with self.w.state.tx() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM deletion_jobs").fetchone()[0], 1)

    def test_rejected_case_releases_without_deletion(self):
        case = self.open_case()
        self.assertEqual(self.w.review.reject(self.P["alice"], case)["status"], "RELEASED")
        self.assertEqual(self.w.state.quarantined_ids(self.w.reg, self.P["proc"]), set())
        with self.assertRaises(AZ.Forbidden):
            self.w.exec.run(self.P["proc"], self.v["cid"])
        self.assertEqual(len(self.objects(self.v["pseud"])), 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
