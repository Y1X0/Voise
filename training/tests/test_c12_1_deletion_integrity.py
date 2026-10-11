"""C12.1 regression tests: write-after-withdrawal, verifiable deletion, path identity, crash-safe
writes, audit chain, account lifecycle (docs/AUTHORIZATION_DESIGN.md §8).

Synthetic only: throw-away principals, fake bytes standing in for audio, temporary directories.
No real audio, no Telegram, no network, no real keys. Each finding from the independent
read-only probe report (P1-P6) has at least one test named after it.

Crash scenarios that must not run any `finally`/compensation code use a forked child that dies
with `os._exit` at a named crash point (Linux/POSIX), so the on-disk state is what a real
process death leaves behind.
"""
import os
import shutil
import sqlite3
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_c12_authorization as T  # noqa: E402
from datasets import audit_chain as AC  # noqa: E402
from datasets import authz as AZ  # noqa: E402
from datasets import secure_state as SS  # noqa: E402


def crash_in_child(fn, point):
    """Run fn in a forked child that dies (no cleanup at all) when it reaches crash point `point`.
    Returns 0 if the child died exactly there."""
    pid = os.fork()
    if pid == 0:                                                   # child
        code = 3
        try:
            SS._crashpoint = lambda name: os._exit(0) if name == point else None
            fn()
            code = 2                                               # crash point never reached
        except BaseException:
            code = 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


class C121(T.Base):
    def restart(self):
        self.w.close()
        self.w = T.World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)

    def withdraw(self, who):
        return self.w.wd.by_token(who["token"])

    def run_exec(self, who):
        return self.w.exec.run(self.P["proc"], who["cid"])

    def dstatus(self, who, principal="wd"):
        return SS.deletion_status(self.w.reg, self.w.state, self.P[principal], who["cid"])

    def pdir(self, who):
        return os.path.join(self.w.store.root, who["pseud"].replace(":", "_"))

    def entries(self, who):
        d = self.pdir(who)
        return sorted(os.listdir(d)) if os.path.isdir(d) else []

    def asset_states(self, who):
        with self.w.state.tx() as c:
            return sorted(r[0] for r in c.execute("SELECT state FROM assets WHERE pseudonym=?", (who["pseud"],)))

    def write_attempts(self, who, tag):
        """Every raw and derived write path, each as a callable."""
        return [
            ("store_audio", lambda: self.w.writer.store_audio(who["pseud"], f"r-{tag}.ogg", b"late raw")),
            ("put", lambda: self.w.store.put(self.P["col"], who["pseud"], f"p-{tag}.ogg", b"late raw")),
            ("put_derived", lambda: self.w.store.put_derived(self.P["proc"], who["pseud"], f"derived-{tag}", b"late")),
            ("put_derived_indexed", lambda: self.w.store.put_derived(self.P["proc"], who["pseud"], f"derived-i-{tag}",
                                                                     b"late", kind="features")),
        ]


# P3 -------------------------------------------------------------- no write after withdrawal / deletion
class WriteAfterWithdrawal(C121):
    def assert_all_writes_refused(self, who, tag):
        before = self.entries(who)
        for name, fn in self.write_attempts(who, tag):
            with self.subTest(path=name), self.assertRaises(AZ.Forbidden):
                fn()
        self.assertEqual(self.entries(who), before)

    def test_P3_every_write_path_refused_after_withdrawal(self):
        self.withdraw(self.v)
        self.assert_all_writes_refused(self.v, "after-withdrawal")
        for name, fn in self.write_attempts(self.a, "other"):              # other participant unaffected
            with self.subTest(other=name):
                fn()

    def test_P3_refused_after_done_and_after_restart(self):
        self.withdraw(self.v)
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")
        self.assert_all_writes_refused(self.v, "after-done")
        self.restart()
        self.assert_all_writes_refused(self.v, "after-restart")
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")
        self.assertEqual(self.dstatus(self.v), "done")

    def test_writes_refused_while_quarantined_and_for_unknown_pseudonym(self):
        self.open_case()
        self.assert_all_writes_refused(self.v, "quarantined")
        ghost = {"pseud": "own_recordings:spk-0123456789"}                  # valid form, no consent
        for name, fn in self.write_attempts(ghost, "ghost"):
            with self.subTest(path=name), self.assertRaises(AZ.Forbidden):
                fn()

    def test_concurrent_writers_vs_withdrawal_and_deletion(self):
        started, stop_errors, lock = threading.Event(), [], threading.Lock()
        ok = {"n": 0}
        refused = []

        def writer(tid):
            w = T.World(self.tmp, self.clock, kms=self.w.kms).bind(self.P)
            i = 0
            try:
                while True:
                    try:
                        if tid % 2:
                            w.writer.store_audio(self.v["pseud"], f"c{tid}-{i}.ogg", b"x" * 256)
                        else:
                            w.store.put_derived(self.P["proc"], self.v["pseud"], f"derived-c{tid}-{i}", b"y" * 256,
                                                kind="features")
                    except AZ.Forbidden:
                        with lock:
                            refused.append(tid)
                        return
                    with lock:
                        ok["n"] += 1
                        if ok["n"] >= 16:
                            started.set()
                    i += 1
                    if i > 5000:
                        raise RuntimeError("writer never refused")
            except BaseException as e:                                       # surfaced below
                with lock:
                    stop_errors.append(repr(e))
            finally:
                w.close()

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        self.assertTrue(started.wait(60))
        self.withdraw(self.v)
        first = self.run_exec(self.v)                                        # writers may still be running
        for t in threads:
            t.join(60)
        self.assertEqual(stop_errors, [])
        self.assertEqual(sorted(refused), list(range(8)))                    # every writer was refused
        self.assertIn(first["status"], ("DONE", "INCOMPLETE"))
        final = self.run_exec(self.v)
        self.assertEqual(final["status"], "DONE", final)
        self.assertEqual(self.entries(self.v), [])
        self.assertEqual(set(self.asset_states(self.v)), {"deleted"})
        # ordering proof from the audit chain: no object was written after the withdrawal committed
        rows = SS.read_audit(self.w.reg, self.w.state, self.P["audit"])
        subj = SS.audit_subject(self.v["cid"])
        w_seq = [r["seq"] for r in rows if r["event"] == "withdrawn" and r["subject"] == subj]
        written = [r["seq"] for r in rows if r["event"] == "object_written" and r["subject"] == subj]
        self.assertEqual(len(w_seq), 1)
        self.assertGreaterEqual(len(written), 16)
        self.assertTrue(all(s < w_seq[0] for s in written), (w_seq, max(written)))
        self.assertTrue(SS.verify_audit(self.w.reg, self.w.state, self.P["audit"])["ok"])


# stale DONE ------------------------------------------------------ completion is re-verified every time
class StaleDone(C121):
    def test_P3_planted_object_after_done_is_never_reported_done(self):
        self.withdraw(self.v)
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")
        os.makedirs(self.pdir(self.v), exist_ok=True)
        with open(os.path.join(self.pdir(self.v), "resurrected.ogg"), "wb") as f:   # e.g. lost unlink after power loss
            f.write(b"came back")
        self.assertEqual(self.dstatus(self.v), "incomplete")                 # status API re-verifies
        self.assertEqual(self.dstatus(self.v, "audit"), "incomplete")
        out = self.run_exec(self.v)
        self.assertEqual((out["status"], out["unindexed_objects"]), ("INCOMPLETE", 1))
        self.restart()
        self.assertEqual(self.dstatus(self.v), "incomplete")                 # persisted downgrade survives restart
        os.remove(os.path.join(self.pdir(self.v), "resurrected.ogg"))
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")

    def test_partial_write_left_after_done_is_detected_then_cleaned(self):
        self.withdraw(self.v)
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")
        os.makedirs(self.pdir(self.v), exist_ok=True)
        with open(os.path.join(self.pdir(self.v), ".tmp-deadbeefdeadbeef"), "wb") as f:   # stalled writer's temp
            f.write(b"half")
        self.assertEqual(self.dstatus(self.v), "incomplete")
        out = self.run_exec(self.v)
        self.assertEqual((out["status"], out["partial_writes_removed"]), ("DONE", 1))
        self.assertEqual(self.entries(self.v), [])

    def test_done_result_states_its_scope(self):
        self.withdraw(self.v)
        out = self.run_exec(self.v)
        self.assertEqual(out["status"], "DONE")
        self.assertIn("object store", out["verified_scope"])
        for item in ("backups", "Telegram", "models", "outside the object store"):
            self.assertTrue(any(item in x for x in out["not_deletable"]), item)


# P4b ------------------------------------------------------------- missing registered objects are not deletions
class MissingObjects(C121):
    def test_P4b_registered_object_moved_out_is_unverified(self):
        outside = os.path.join(self.tmp, "elsewhere")
        os.makedirs(outside)
        derived = [p for p in self.objects(self.v["pseud"]) if "derived" in p][0]
        shutil.move(derived, os.path.join(outside, "moved.npy"))
        self.withdraw(self.v)
        out = self.run_exec(self.v)
        self.assertEqual((out["status"], out["missing_assets"]), ("UNVERIFIED", 1))
        self.assertEqual(self.dstatus(self.v), "unverified")
        self.assertEqual(self.run_exec(self.v)["status"], "UNVERIFIED")      # retry does not upgrade it
        self.restart()
        self.assertEqual(self.run_exec(self.v)["status"], "UNVERIFIED")
        self.assertTrue(os.path.exists(os.path.join(outside, "moved.npy")))  # out of scope: untouched, not claimed

    def test_missing_plus_remaining_object_is_incomplete(self):
        derived = [p for p in self.objects(self.v["pseud"]) if "derived" in p][0]
        os.remove(derived)
        os.makedirs(self.pdir(self.v), exist_ok=True)
        with open(os.path.join(self.pdir(self.v), "stray.bin"), "wb") as f:
            f.write(b"x")
        self.withdraw(self.v)
        self.assertEqual(self.run_exec(self.v)["status"], "INCOMPLETE")

    def test_crash_after_unlink_before_record_is_idempotent(self):
        self.withdraw(self.v)
        code = crash_in_child(lambda: self.run_exec(self.v), "delete.after_unlink")
        self.assertEqual(code, 0)
        self.assertEqual(self.dstatus(self.v), "in_progress")
        self.assertIn("deleting", self.asset_states(self.v))
        self.restart()
        out = self.run_exec(self.v)
        self.assertEqual(out["status"], "DONE", out)                         # our own recorded intent, not "missing"
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")
        self.assertEqual(set(self.asset_states(self.v)), {"deleted"})


# P4a / paths ------------------------------------------------------ identity and containment of every reference
class PathIdentity(C121):
    def register(self, who, path, principal="proc"):
        return SS.register_derived(self.w.reg, self.w.state, self.P[principal], who["pseud"], "features", path)

    def test_P4a_registration_outside_root_refused(self):
        outside = os.path.join(self.tmp, "elsewhere")
        os.makedirs(outside)
        p = os.path.join(outside, "derived-copy.npy")
        with open(p, "wb") as f:
            f.write(b"x")
        before = self.asset_states(self.v)
        for bad in (p, "/etc/passwd", "relative/derived.npy", "", None, 42):
            with self.subTest(path=bad), self.assertRaises(AZ.Forbidden):
                self.register(self.v, bad)
        self.assertEqual(self.asset_states(self.v), before)
        self.withdraw(self.v)
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")            # not stuck in_progress
        self.assertTrue(os.path.exists(p))

    def test_registration_of_other_participant_or_traversal_refused(self):
        victim_file = [p for p in self.objects(self.v["pseud"]) if "derived" in p][0]
        with self.assertRaises(AZ.Forbidden):
            self.register(self.a, victim_file)                               # victim's file under attacker's pseudonym
        trav = os.path.join(self.pdir(self.a), "..", os.path.basename(self.pdir(self.v)), os.path.basename(victim_file))
        with self.assertRaises(AZ.Forbidden):
            self.register(self.a, trav)
        with self.assertRaises(AZ.Forbidden):
            self.register(self.v, os.path.join(self.pdir(self.v), "not-there.npy"))   # must exist
        self.register(self.v, victim_file)                                   # idempotent for the owner

    def test_symlink_object_escape_refused(self):
        outside = os.path.join(self.tmp, "secret.txt")
        with open(outside, "wb") as f:
            f.write(b"outside the store")
        link = os.path.join(self.pdir(self.v), "derived-link")
        os.symlink(outside, link)
        with self.assertRaises(AZ.Forbidden):
            self.register(self.v, link)
        with self.assertRaises(AZ.Forbidden):
            self.w.store.get(self.P["proc"], self.v["pseud"], "derived-link")
        self.withdraw(self.v)
        out = self.run_exec(self.v)
        self.assertEqual((out["status"], out["unindexed_objects"]), ("INCOMPLETE", 1))   # flagged, not followed
        with open(outside, "rb") as f:
            self.assertEqual(f.read(), b"outside the store")

    def test_symlinked_participant_directory_refused(self):
        a_dir, v_dir = self.pdir(self.a), self.pdir(self.v)
        shutil.rmtree(a_dir)
        os.symlink(v_dir, a_dir)                                             # attacker dir aliases victim dir
        self.w.wd.by_token(self.a["token"])
        for fn in (lambda: self.w.store.list(self.P["proc"], self.a["pseud"]),
                   lambda: self.w.store.get(self.P["proc"], self.a["pseud"], "p1.ogg"),
                   lambda: self.w.store.delete(self.P["proc"], self.a["pseud"], os.path.join(a_dir, "p1.ogg")),
                   lambda: self.w.store.put_derived(self.P["proc"], self.a["pseud"], "derived-x", b"x")):
            with self.subTest(fn=fn), self.assertRaises(AZ.Forbidden):
                fn()
        with self.assertRaises(AZ.Forbidden):
            self.run_exec(self.a)
        self.assertEqual(self.dstatus(self.a), "in_progress")                # never done through an alias
        self.assertEqual(len(self.objects(self.v["pseud"])), 2)              # victim untouched

    def test_corrupt_index_row_is_unverified_not_stuck(self):
        outside = os.path.join(self.tmp, "x.npy")
        with open(outside, "wb") as f:
            f.write(b"x")
        con = sqlite3.connect(self.w.state.path)                             # direct DB edit (no API path allows it)
        con.execute("INSERT INTO assets VALUES ('bad1', ?, 'features', ?, 'registered')", (self.v["pseud"], outside))
        con.commit()
        con.close()
        self.withdraw(self.v)
        out = self.run_exec(self.v)
        self.assertEqual(out["status"], "UNVERIFIED")
        self.assertTrue(os.path.exists(outside))

    def test_store_and_state_are_bound(self):
        with self.assertRaises((AZ.Forbidden, AZ.StoreUnavailable)):
            SS.ObjectStore(self.w.reg, os.path.join(self.tmp, "other-root"), self.w.state)


# crash-safe writes --------------------------------------------------------------------------
class CrashSafeWrites(C121):
    def test_partial_write_never_becomes_an_object(self):
        code = crash_in_child(lambda: self.w.writer.store_audio(self.v["pseud"], "p2.ogg", b"A" * 4096), "write.mid")
        self.assertEqual(code, 0)
        self.assertNotIn("p2.ogg", self.entries(self.v))
        tmps = [e for e in self.entries(self.v) if e.startswith(".tmp-")]
        self.assertEqual(len(tmps), 1)
        self.restart()
        self.assertEqual(SS.ObjectStore.recover(self.w.store, self.P["proc"], self.v["pseud"]), 1)
        self.assertEqual([e for e in self.entries(self.v) if e.startswith(".tmp-")], [])
        self.w.writer.store_audio(self.v["pseud"], "p2.ogg", b"A" * 4096)  # retry with the same name works
        self.assertEqual(self.w.store.get(self.P["proc"], self.v["pseud"], "p2.ogg"), b"A" * 4096)
        with self.assertRaises(AZ.Forbidden):
            self.w.store.recover(self.P["col"], self.v["pseud"])

    def test_P5_crash_between_link_and_commit_is_detected(self):
        code = crash_in_child(lambda: self.w.writer.store_audio(self.v["pseud"], "p3.ogg", b"B" * 64), "write.after_link")
        self.assertEqual(code, 0)
        self.restart()
        self.assertIn("p3.ogg", self.entries(self.v))                       # linked, but index row rolled back
        self.withdraw(self.v)
        out = self.run_exec(self.v)
        self.assertEqual((out["status"], out["unindexed_objects"]), ("INCOMPLETE", 1))   # never a false DONE
        os.remove(os.path.join(self.pdir(self.v), "p3.ogg"))                  # operator resolves the lineage gap
        self.assertEqual(self.run_exec(self.v)["status"], "DONE")

    def test_partial_write_of_withdrawn_participant_removed_by_deletion(self):
        code = crash_in_child(lambda: self.w.store.put_derived(self.P["proc"], self.v["pseud"], "derived-z", b"Z" * 4096),
                              "write.mid")
        self.assertEqual(code, 0)
        self.restart()
        self.withdraw(self.v)
        out = self.run_exec(self.v)
        self.assertEqual((out["status"], out["partial_writes_removed"]), ("DONE", 1))

    def test_old_schema_state_db_refused(self):
        old = os.path.join(self.tmp, "old.sqlite")
        con = sqlite3.connect(old)
        con.execute("CREATE TABLE consents (consent_id TEXT PRIMARY KEY, pseudonym TEXT, token_sha256 TEXT, status TEXT)")
        con.commit()
        con.close()
        with self.assertRaises(AZ.StoreUnavailable):
            SS.StateDB(old)


# audit chain ------------------------------------------------------------------------------------
class AuditChain(C121):
    def tamper(self, sql):
        con = sqlite3.connect(self.w.state.path)
        con.executescript("DROP TRIGGER audit_no_update; DROP TRIGGER audit_no_delete;" + sql +
                          "CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT,'x'); END;"
                          "CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT,'x'); END;")
        con.close()

    def verify(self, anchor=None):
        return SS.verify_audit(self.w.reg, self.w.state, self.P["audit"], anchor=anchor)

    def setUp(self):
        super().setUp()
        self.withdraw(self.v)
        self.run_exec(self.v)
        self.assertGreaterEqual(len(SS.read_audit(self.w.reg, self.w.state, self.P["audit"])), 6)

    def test_clean_chain_verifies(self):
        r = self.verify()
        self.assertTrue(r["ok"], r)
        self.assertTrue(self.verify(anchor=r["head"])["ok"])

    def test_P2_deleted_row_detected(self):
        self.tamper("DELETE FROM audit WHERE seq=2;")
        r = self.verify()
        self.assertFalse(r["ok"])
        self.assertEqual(r["bad_seq"], 3)

    def test_mutation_detected(self):
        self.tamper("UPDATE audit SET event='nothing_happened' WHERE seq=3;")
        self.assertFalse(self.verify()["ok"])

    def test_reordering_detected(self):
        self.tamper("UPDATE audit SET seq=-1 WHERE seq=2; UPDATE audit SET seq=2 WHERE seq=3; UPDATE audit SET seq=3 WHERE seq=-1;")
        self.assertFalse(self.verify()["ok"])

    def test_truncation_and_recomputation_need_an_external_anchor(self):
        anchor = self.verify()["head"]
        self.tamper(f"DELETE FROM audit WHERE seq={anchor['seq']};")        # drop the newest row
        self.assertTrue(self.verify()["ok"])                                  # stated limit: undetectable alone
        self.assertFalse(self.verify(anchor=anchor)["ok"])
        # a DB-file administrator rewrites the whole chain consistently
        self.restart()
        self.tamper("DELETE FROM audit WHERE seq=2;")
        con = sqlite3.connect(self.w.state.path)
        con.executescript("DROP TRIGGER audit_no_update; DROP TRIGGER audit_no_delete;")
        prev = AC.GENESIS
        rows = con.execute("SELECT seq, " + ", ".join(SS.AUDIT_COLUMNS) + " FROM audit ORDER BY seq").fetchall()
        con.execute("DELETE FROM audit")
        for i, r in enumerate(rows, 1):                                      # renumber + recompute (no secret needed)
            h = AC.row_hash(prev, i, list(r[1:]))
            con.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?,?,?)", (i, *r[1:], prev, h))
            prev = h
        con.commit()
        con.close()
        self.assertTrue(self.verify()["ok"])                                  # consistent forgery passes alone
        self.assertFalse(self.verify(anchor=anchor)["ok"])                    # but not against the anchor

    def test_verify_needs_auditor(self):
        for name, p in self.P.items():
            if name == "audit":
                continue
            with self.subTest(name=name), self.assertRaises(AZ.Forbidden):
                SS.verify_audit(self.w.reg, self.w.state, p)


# P1 / P6 -------------------------------------------------------- account lifecycle
class Accounts(C121):
    def test_P6_unauthenticated_account_changes_refused_once_sealed(self):
        self.w.reg.seal(self.P["admin"])
        for fn in (lambda: self.w.reg.provision("rogue", AZ.AUDITOR, "nobody"),
                   lambda: AZ.Registry(self.w.reg.path).provision("rogue", AZ.AUDITOR, "nobody"),
                   lambda: self.w.reg.deactivate("staff-bob")):
            with self.subTest(fn=fn), self.assertRaises(AZ.Forbidden):
                fn()
        self.w.reg.authorize(self.P["bob"], "case.read")                    # bob still active

    def test_admin_lifecycle_is_authorized_and_audited(self):
        self.w.reg.seal(self.P["admin"])
        erin = self.w.reg.admin_provision(self.P["admin"], "staff-erin", AZ.REVIEWER, "person-erin")
        self.w.reg.authorize(erin, "case.read")
        self.w.reg.admin_deactivate(self.P["admin"], "staff-erin")
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.authorize(erin, "case.read")
        rows = AZ.read_account_audit(self.w.reg, self.P["audit"])
        events = [(r["event"], r["target"]) for r in rows]
        self.assertIn(("sealed", None), events)
        self.assertIn(("provisioned", "staff-erin"), events)
        self.assertIn(("deactivated", "staff-erin"), events)
        text = repr(rows)
        for p in list(self.P.values()) + [erin]:
            self.assertNotIn(p._credential, text)
        self.assertNotIn("person-erin", text)                                # staff person referenced by hash only
        self.assertTrue(AZ.verify_account_audit(self.w.reg, self.P["audit"])["ok"])

    def test_non_admins_cannot_change_accounts(self):
        for name, p in self.P.items():
            if name == "admin":
                continue
            with self.subTest(name=name):
                with self.assertRaises(AZ.Forbidden):
                    self.w.reg.admin_provision(p, f"x-{name}", AZ.APPROVER, f"person-x{name}")
                with self.assertRaises(AZ.Forbidden):
                    self.w.reg.admin_deactivate(p, "staff-bob")

    def test_admin_separation(self):
        adm = self.P["admin"]
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.admin_provision(adm, "admin-2", AZ.ACCOUNT_ADMIN, "person-zoe")   # no admin minting
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.admin_provision(adm, "dana-reviewer", AZ.REVIEWER, "person-dana")  # not for themself
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.admin_deactivate(adm, "staff-admin")
        with self.assertRaises(AZ.Forbidden):
            self.w.reg.admin_provision(adm, "staff-alice", AZ.APPROVER, "person-q")       # no takeover of an id

    def test_account_audit_tamper_detected(self):
        self.w.reg.admin_provision(self.P["admin"], "staff-erin", AZ.REVIEWER, "person-erin")
        con = sqlite3.connect(self.w.reg.path)
        con.executescript("DROP TRIGGER account_audit_no_delete; DELETE FROM account_audit WHERE seq=2;")
        con.close()
        self.assertFalse(AZ.verify_account_audit(self.w.reg, self.P["audit"])["ok"])

    def test_P1_local_person_ref_is_reported_unverified(self):
        """The registry cannot prove that two person_refs are two humans; it says so instead of
        pretending, and rejects trivially varied spellings."""
        x1 = self.w.reg.provision("mallory-a", AZ.REVIEWER, "person-mallory")
        x2 = self.w.reg.provision("mallory-b", AZ.APPROVER, "person-m4llory")
        case = self.open_case()
        self.assertEqual(self.w.review.approve(x1, case)["identity_assurance"], AZ.IDENTITY_ASSURANCE)
        out = self.w.review.approve(x2, case)
        self.assertEqual(out["identity_assurance"], "local-unverified")      # production blocker, not a guarantee
        for bad in ("Person-Alice", "person-alice ", " person-alice", "person alice", ""):
            with self.subTest(ref=bad), self.assertRaises(ValueError):
                self.w.reg.provision(f"dup-{abs(hash(bad))}", AZ.APPROVER, bad)


if __name__ == "__main__":
    unittest.main(verbosity=1)
