"""Synthetic tests for C5/C6 (docs/KEY_MANAGEMENT_DESIGN.md): encryption to operational + escrow
keys, pinned public keys, signed hash-chained object log, tamper / injection / rotation / loss.

Synthetic data only (generated tones), throw-away keys made in a temp dir. No network.
  python3 training/tests/test_secure_store.py
"""
import base64
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from datasets import secure_store as SS  # noqa: E402

NEED_AGE = unittest.skipUnless(SS.age_available(), "age / age-keygen CLI not installed")


def tone(seed):
    t = np.arange(16000) / 16000
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * (150 + 20 * seed) * t)).astype(np.float32), 16000, format="OGG",
             subtype="VORBIS")
    return buf.getvalue()


def keygen(path):
    subprocess.run(["age-keygen", "-o", path], check=True, capture_output=True)
    return subprocess.run(["age-keygen", "-y", path], check=True, capture_output=True, text=True).stdout.strip()


@NEED_AGE
class KeysAndEncryption(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.op_id, self.esc_id, self.other_id = (os.path.join(self.tmp, n) for n in ("op.key", "esc.key", "x.key"))
        self.op, self.esc, self.other = keygen(self.op_id), keygen(self.esc_id), keygen(self.other_id)
        self.cfg = {"operational": {"public": self.op, "key_id": SS.key_id(self.op)},
                    "escrow": {"public": self.esc, "key_id": SS.key_id(self.esc)}}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_roundtrip_both_recipients_and_third_key_fails(self):
        data = tone(1)
        ct = SS.encrypt(data, [self.op, self.esc])
        self.assertNotIn(data[:64], ct)
        self.assertEqual(SS.decrypt(ct, self.op_id), data)
        self.assertEqual(SS.decrypt(ct, self.esc_id), data)          # key-loss recovery path
        with self.assertRaises(SS.SecureStoreError):
            SS.decrypt(ct, self.other_id)
        self.assertEqual(SS.recipient_count(ct), 2)

    def test_single_recipient_or_malformed_key_refused(self):
        with self.assertRaises(SS.SecureStoreError):
            SS.encrypt(b"x", [self.op])
        with self.assertRaises(SS.SecureStoreError):
            SS.encrypt(b"x", [self.op, "age1notakey"])

    def test_pinned_keys(self):
        self.assertEqual(SS.load_pinned_recipients(self.cfg)["escrow"], self.esc)
        swapped = {**self.cfg, "escrow": {"public": self.other, "key_id": SS.key_id(self.esc)}}
        with self.assertRaises(SS.SecureStoreError):
            SS.load_pinned_recipients(swapped)                        # attacker key with old pin
        same = {**self.cfg, "escrow": dict(self.cfg["operational"])}
        with self.assertRaises(SS.SecureStoreError):
            SS.load_pinned_recipients(same)

    def test_collector_config_must_hold_no_private_keys(self):
        cfgdir = os.path.join(self.tmp, "collector_cfg")
        os.makedirs(cfgdir)
        with open(os.path.join(cfgdir, "recipients.txt"), "w") as f:
            f.write(self.op + "\n" + self.esc + "\n")
        SS.assert_no_private_material(cfgdir)
        shutil.copy(self.op_id, os.path.join(cfgdir, "oops.key"))
        with self.assertRaises(SS.SecureStoreError):
            SS.assert_no_private_material(cfgdir)

    def test_tampered_ciphertext_fails_decrypt(self):
        ct = bytearray(SS.encrypt(tone(2), [self.op, self.esc]))
        ct[-5] ^= 0x01
        with self.assertRaises(SS.SecureStoreError):
            SS.decrypt(bytes(ct), self.op_id)

    def test_rotation_rewrap(self):
        data = tone(3)
        ct = SS.encrypt(data, [self.op, self.esc])
        new_id = os.path.join(self.tmp, "op2.key")
        new_op = keygen(new_id)
        ct2 = SS.rewrap(ct, self.op_id, [new_op, self.esc])
        self.assertEqual(SS.decrypt(ct2, new_id), data)
        self.assertEqual(SS.decrypt(ct2, self.esc_id), data)
        with self.assertRaises(SS.SecureStoreError):
            SS.decrypt(ct2, self.op_id)                                # retired key no longer opens new copy


@NEED_AGE
class SignedObjectLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.op_id, self.esc_id = os.path.join(self.tmp, "op.key"), os.path.join(self.tmp, "esc.key")
        op, esc = keygen(self.op_id), keygen(self.esc_id)
        self.recipients = {"operational": op, "escrow": esc}
        self.required = [SS.key_id(op), SS.key_id(esc)]
        self.col_key, self.proc_key = SS.new_signing_key(), SS.new_signing_key()
        self.trusted = {SS.key_id(SS.ed25519_public_b64(self.col_key)): (self.col_key.public_key(), "collector"),
                        SS.key_id(SS.ed25519_public_b64(self.proc_key)): (self.proc_key.public_key(), "processing")}
        self.objects = {}
        self.log = SS.SignedLog(self.col_key, "collector")
        self.writer = SS.CollectorWriter(self.recipients, self.log, self.objects.__setitem__)
        for i in range(3):
            self.writer.store(f"obj-{i}", "own_recordings:spk-0123456789", f"ses-0000000{i}", tone(i))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def accept(self, oid, entries=None, ct=None):
        acc = SS.verify_log(entries if entries is not None else self.log.entries, self.trusted)
        return SS.accept_object(oid, ct if ct is not None else self.objects[oid], acc, self.required)

    def test_valid_objects_accepted_and_decryptable(self):
        for oid in self.objects:
            self.accept(oid)
            self.assertTrue(SS.decrypt(self.objects[oid], self.op_id).startswith(b"OggS"))

    def test_ciphertext_swap_or_tamper_detected(self):
        with self.assertRaises(SS.SecureStoreError):
            self.accept("obj-0", ct=self.objects["obj-1"])
        bad = bytearray(self.objects["obj-0"])
        bad[-1] ^= 1
        with self.assertRaises(SS.SecureStoreError):
            self.accept("obj-0", ct=bytes(bad))

    def test_injected_object_without_signed_entry_rejected(self):
        forged = SS.encrypt(tone(9), [self.recipients["operational"], self.recipients["escrow"]])  # public keys suffice
        with self.assertRaises(SS.SecureStoreError):
            self.accept("obj-forged", ct=forged)

    def test_entry_signed_by_unknown_key_rejected(self):
        rogue = SS.SignedLog(SS.new_signing_key(), "collector")
        rogue.entries = [dict(e) for e in self.log.entries]
        rogue.append({"kind": "store", "object_id": "obj-x", "pseudonym": "p", "session_id": "s",
                      "ciphertext_sha256": "0" * 64, "recipient_key_ids": sorted(self.required), "recipient_count": 2})
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log(rogue.entries, self.trusted)

    def test_modified_entry_deleted_or_reordered_entries_detected(self):
        e = [dict(x) for x in self.log.entries]
        e[1]["pseudonym"] = "own_recordings:spk-aaaaaaaaaa"
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log(e, self.trusted)                              # bad signature
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log([self.log.entries[0], self.log.entries[2]], self.trusted)   # deletion
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log([self.log.entries[1], self.log.entries[0], self.log.entries[2]], self.trusted)

    def test_tail_truncation_needs_anchored_head(self):
        anchored, n = self.log.head(), len(self.log.entries)
        cut = self.log.entries[:-1]
        SS.verify_log(cut, self.trusted)                         # limitation: chain alone cannot see a cut tail
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log(cut, self.trusted, expected_head=anchored)
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log(cut, self.trusted, min_entries=n)
        SS.verify_log(self.log.entries, self.trusted, expected_head=anchored, min_entries=n)

    def test_missing_escrow_recipient_rejected(self):
        log = SS.SignedLog(self.col_key, "collector")
        third = keygen(os.path.join(self.tmp, "third.key"))
        ct = SS.encrypt(tone(5), [self.recipients["operational"], third])          # escrow left out
        log.append({"kind": "store", "object_id": "obj-e", "pseudonym": "p", "session_id": "s",
                    "ciphertext_sha256": hashlib.sha256(ct).hexdigest(),
                    "recipient_key_ids": sorted([SS.key_id(self.recipients["operational"]), SS.key_id(third)]),
                    "recipient_count": 2})
        with self.assertRaises(SS.SecureStoreError):
            SS.accept_object("obj-e", ct, SS.verify_log(log.entries, self.trusted), self.required)

    def test_role_separation_in_log(self):
        e = [dict(x) for x in self.log.entries]
        plog = SS.SignedLog(self.proc_key, "processing")
        plog.entries = e
        plog.append({"kind": "store", "object_id": "obj-p", "pseudonym": "p", "session_id": "s",
                     "ciphertext_sha256": "0" * 64, "recipient_key_ids": sorted(self.required), "recipient_count": 2})
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log(plog.entries, self.trusted)                   # processing host cannot create objects
        clog = SS.SignedLog(self.col_key, "collector")
        clog.entries = [dict(x) for x in self.log.entries]
        clog.append({"kind": "delete", "object_id": "obj-0"})
        with self.assertRaises(SS.SecureStoreError):
            SS.verify_log(clog.entries, self.trusted)                   # collector cannot record deletions

    def test_rotation_and_deletion_recorded_by_processing_host(self):
        new_id = os.path.join(self.tmp, "op2.key")
        new_op = keygen(new_id)
        plog = SS.SignedLog(self.proc_key, "processing")
        plog.entries = [dict(x) for x in self.log.entries]
        ct2 = SS.rewrap(self.objects["obj-0"], self.op_id, [new_op, self.recipients["escrow"]])
        new_required = [SS.key_id(new_op), SS.key_id(self.recipients["escrow"])]
        plog.append({"kind": "rewrap", "object_id": "obj-0", "ciphertext_sha256": hashlib.sha256(ct2).hexdigest(),
                     "recipient_key_ids": sorted(new_required), "recipient_count": 2,
                     "replaces_sha256": hashlib.sha256(self.objects["obj-0"]).hexdigest()})
        acc = SS.verify_log(plog.entries, self.trusted)
        SS.accept_object("obj-0", ct2, acc, new_required)
        with self.assertRaises(SS.SecureStoreError):
            SS.accept_object("obj-0", self.objects["obj-0"], acc, new_required)  # old copy no longer valid
        plog.append({"kind": "delete", "object_id": "obj-1", "reason": "withdrawal"})
        acc = SS.verify_log(plog.entries, self.trusted)
        with self.assertRaises(SS.SecureStoreError):
            SS.accept_object("obj-1", self.objects["obj-1"], acc, self.required)

    def test_signatures_are_ed25519_and_canonical(self):
        e = self.log.entries[0]
        body = {k: v for k, v in e.items() if k != "sig"}
        self.col_key.public_key().verify(base64.b64decode(e["sig"]), SS.canonical(body))
        self.assertEqual(len(base64.b64decode(e["sig"])), 64)


if __name__ == "__main__":
    unittest.main(verbosity=1)
