#!/usr/bin/env python3
"""Reproducible manifest builder (B4): local corpus layouts -> speaker/session-disjoint
splits -> leakage check (fail-fast) -> hashed JSONL manifests + statistics report.

Nothing is downloaded. Each builder reads an already-downloaded corpus in its official
layout and emits rows in the manifest format (datasets/manifest.py):

  librispeech   <root>/<subset>/<spk>/<chapter>/<spk>-<chapter>-<n>.flac + <spk>-<chapter>.trans.txt
                (also LibriTTS / LibriTTS-R: <utt>.wav + <utt>.normalized.txt); session = chapter
  vctk          <root>/wav48_silence_trimmed/<spk>/<utt>_mic1.flac + <root>/txt/<spk>/<utt>.txt;
                session unknown -> "single" (VCTK speakers cannot be cross-session TEST speakers)
  commonvoice   <root>/<lang>/validated.tsv (client_id, path, sentence) + clips/; session unknown
  segments      a TSV  path<TAB>speaker<TAB>session<TAB>start<TAB>end<TAB>text  (e.g. AMI IHM
                segments derived from the official annotations; session = meeting)
  generic       <root>/<speaker>/<session>/<file>.wav (+ same-name .txt)  (own recordings;
                <root>/<speaker>/consent_id.txt links the speaker to a consent record)

Splitting (deterministic given --seed):
  * excluded speakers / reserved subsets come from datasets/excluded_speakers.json;
  * reserved attacker subsets (e.g. librispeech train-clean-360) -> split attacker_train;
  * remaining speakers are shuffled per corpus and assigned by fractions to train / valid / test;
  * TEST speakers must have >= 2 sessions; their utterances get role=enroll (first session) or
    role=trial (other sessions): session-disjoint enrollment/trial;
  * then datasets/leakage.check + session checks run, and ANY finding aborts (exit 1).
Outputs in --out: train.jsonl valid.jsonl test.jsonl attacker_train.jsonl (sorted),
manifest_index.json (sha256 per manifest, seed, inputs), stats.json, stats.md.
`--verify DIR` re-checks hashes + leakage of an existing directory (exit 1 on any finding).

  python3 training/datasets/build_manifests.py --out data/manifests --seed 2026 \
      --librispeech /data/LibriTTS_R:librittsr:train-clean-100,train-other-500 \
      --librispeech /data/LibriSpeech:librispeech:train-clean-360 \
      --vctk /data/VCTK-Corpus-0.92 --segments /data/ami_ihm_segments.tsv:ami
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from datasets.leakage import canonical, check as leakage_check, load_exclusions, reserved_splits  # noqa: E402
from datasets.manifest import validate  # noqa: E402

AUDIO = (".wav", ".flac", ".mp3", ".ogg")


class LeakageError(RuntimeError):
    pass


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _dur(path):
    import soundfile as sf
    i = sf.info(path)
    return i.frames / i.samplerate, i.samplerate


def build_librispeech(root, corpus, subsets, language="en"):
    rows = []
    for sub in subsets:
        for spk_dir in sorted(glob.glob(os.path.join(root, sub, "*"))):
            spk = os.path.basename(spk_dir)
            for ch_dir in sorted(glob.glob(os.path.join(spk_dir, "*"))):
                ch = os.path.basename(ch_dir)
                texts = {}
                for tf in glob.glob(os.path.join(ch_dir, "*.trans.txt")):
                    for line in open(tf, encoding="utf-8"):
                        k, _, t = line.strip().partition(" ")
                        texts[k] = t
                for a in sorted(f for f in glob.glob(os.path.join(ch_dir, "*")) if f.endswith(AUDIO)):
                    key = os.path.splitext(os.path.basename(a))[0]
                    txt = texts.get(key)
                    norm = os.path.splitext(a)[0] + ".normalized.txt"
                    if txt is None and os.path.exists(norm):
                        txt = open(norm, encoding="utf-8").read().strip()
                    d, sr = _dur(a)
                    rows.append({"path": a, "speaker": f"{corpus}:{spk}", "session": ch, "corpus": corpus,
                                 "subset": sub, "language": language, "duration": round(d, 3), "sr": sr,
                                 "text": txt, "style": "read"})
    return rows


def build_vctk(root):
    """VCTK 0.92 (wav48_silence_trimmed/<spk>/<utt>_mic1.flac) or, if that tree is absent,
    the older 0.80 release (wav48/<spk>/<utt>.wav, not silence-trimmed)."""
    rows = []
    audio = sorted(glob.glob(os.path.join(root, "wav48_silence_trimmed", "*", "*_mic1.flac")))
    suffix = "_mic1.flac"
    if not audio:
        audio, suffix = sorted(glob.glob(os.path.join(root, "wav48", "*", "*.wav"))), ".wav"
    for a in audio:
        spk = os.path.basename(os.path.dirname(a))
        utt = os.path.basename(a).replace(suffix, "")
        tp = os.path.join(root, "txt", spk, utt + ".txt")
        d, sr = _dur(a)
        rows.append({"path": a, "speaker": f"vctk:{spk}", "session": "single", "corpus": "vctk", "subset": "all",
                     "language": "en", "duration": round(d, 3), "sr": sr,
                     "text": open(tp, encoding="utf-8").read().strip() if os.path.exists(tp) else None,
                     "style": "read"})
    return rows


def build_commonvoice(root, lang):
    rows = []
    with open(os.path.join(root, lang, "validated.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a = os.path.join(root, lang, "clips", r["path"])
            if not os.path.exists(a):
                continue
            d, sr = _dur(a)
            rows.append({"path": a, "speaker": f"commonvoice:{r['client_id']}", "session": "unknown",
                         "corpus": "commonvoice", "subset": "validated", "language": lang, "duration": round(d, 3),
                         "sr": sr, "text": r.get("sentence"), "style": "read"})
    return rows


def build_segments(tsv, corpus, language="en"):
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            path, spk, sess, st, en, *txt = line.rstrip("\n").split("\t")
            _, sr = _dur(path)
            rows.append({"path": path, "speaker": f"{corpus}:{spk}", "session": sess, "corpus": corpus,
                         "subset": "segments", "language": language, "offset": float(st),
                         "duration": round(float(en) - float(st), 3), "sr": sr, "text": txt[0] if txt else None,
                         "style": "spontaneous"})
    return rows


def build_generic(root, corpus, language):
    rows = []
    for a in sorted(f for f in glob.glob(os.path.join(root, "*", "*", "*")) if f.endswith(AUDIO)):
        sess = os.path.basename(os.path.dirname(a))
        spk = os.path.basename(os.path.dirname(os.path.dirname(a)))
        tp = os.path.splitext(a)[0] + ".txt"
        d, sr = _dur(a)
        row = {"path": a, "speaker": f"{corpus}:{spk}", "session": sess, "corpus": corpus, "subset": "all",
               "language": language, "duration": round(d, 3), "sr": sr,
               "text": open(tp, encoding="utf-8").read().strip() if os.path.exists(tp) else None,
               "style": "conversational"}
        cp = os.path.join(root, spk, "consent_id.txt")      # own recordings: per-speaker consent id
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                row["consent_id"] = f.read().strip()
        rows.append(row)
    return rows


def assign_splits(rows, seed, fractions=(0.90, 0.05, 0.05), attacker_subsets=(("librispeech", "train-clean-360"),),
                  excl=None):
    """Speaker-disjoint split; TEST speakers need >= 2 sessions and get session-disjoint
    enroll/trial roles. Excluded speakers and reserved non-attacker subsets are dropped."""
    excl = excl or load_exclusions()
    space = excl["same_speaker_space"]
    banned = {canonical(s, space) for g in excl["speakers"].values() for s in g}
    attack = {(canonical(c + ":x", space).split(":")[0], s) for c, s in attacker_subsets}
    out, dropped = [], defaultdict(int)
    by_spk = defaultdict(list)
    for r in rows:
        corp = canonical(r["corpus"] + ":x", space).split(":")[0]
        key = (corp, r.get("subset", ""))
        if canonical(r["speaker"], space) in banned:
            dropped["excluded_speaker"] += 1
            continue
        forbidden = reserved_splits(r["corpus"], r.get("subset", ""), excl)
        if key in attack and "attacker_train" not in forbidden:
            out.append(dict(r, split="attacker_train"))
            continue
        if forbidden:                       # reserved rows never get a training-side split here
            dropped["reserved_subset"] += 1
            continue
        by_spk[canonical(r["speaker"], space)].append(r)
    attacker_spk = {canonical(r["speaker"], space) for r in out}
    rng = random.Random(seed)
    by_corpus = defaultdict(list)
    for s in sorted(by_spk):
        if s in attacker_spk:
            dropped["speaker_also_in_attacker_pool"] += len(by_spk[s])
            continue
        by_corpus[s.split(":")[0]].append(s)
    for corp, spks in sorted(by_corpus.items()):
        rng.shuffle(spks)
        n = len(spks)
        n_test = int(round(fractions[2] * n))
        n_valid = int(round(fractions[1] * n))
        multi = [s for s in spks if len({r["session"] for r in by_spk[s]}) >= 2]
        test = multi[:n_test]
        rest = [s for s in spks if s not in set(test)]
        valid, train = rest[:n_valid], rest[n_valid:]
        for split, group in (("train", train), ("valid", valid)):
            for s in group:
                out += [dict(r, split=split) for r in by_spk[s]]
        for s in test:
            sessions = sorted({r["session"] for r in by_spk[s]})
            for r in by_spk[s]:
                out.append(dict(r, split="test", role="enroll" if r["session"] == sessions[0] else "trial"))
    return sorted(out, key=lambda r: (r["split"], r["speaker"], r["session"], r["path"])), dict(dropped)


def session_checks(rows):
    errors = []
    enroll, trial = defaultdict(set), defaultdict(set)
    for r in rows:
        if r["split"] == "test":
            (enroll if r.get("role") == "enroll" else trial)[r["speaker"]].add(r["session"])
    for spk in enroll:
        both = enroll[spk] & trial.get(spk, set())
        if both:
            errors.append(f"test speaker {spk}: session(s) {sorted(both)} in both enroll and trial")
        if not trial.get(spk):
            errors.append(f"test speaker {spk}: no trial session (cross-session test impossible)")
    return errors


# ------------------------------------------------------------------ C9 / R10: participant timing
# Policy (docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md R10, C9): the consent store keeps exact
# times; exported manifests carry dates at MONTH precision at most and no consent times. The
# builder never writes a date field, so the remaining channels are identifiers taken from the
# participant's directory tree (session, path, subset) and any extra field. Participant rows
# (own_recordings, or any row carrying a consent_id) are REFUSED, never rewritten: rewriting a path
# or session would break audio loading, the lineage index and deletion. Public corpora are not
# participant data and are not checked (e.g. VoxPopuli session ids embed dates).
#  * ids must have their random schema form: speaker own_recordings:spk-<10 hex>
#    (data/consent_schema.json), session ses-<8 hex> (data/recording_session_schema.json),
#    consent_id CNS-<12 hex>. Ids in these forms are random by schema and are masked before the
#    date scan, so an all-digit hex id is never mistaken for a date;
#  * every other metadata value is scanned for dates/times finer than a month after mapping all
#    Unicode decimal digits (Arabic-Indic, Eastern Arabic-Indic, ...) to ASCII;
#  * `text` is speech content (prompts/transcripts), not metadata, and is NOT scanned.
# The scan is pattern-based: forms it does not list (spelled-out numbers, free wording, relative
# dates, calendars other than Gregorian/Hijri month names) are NOT detected.
PARTICIPANT_FIELDS = {"path", "speaker", "session", "corpus", "language", "duration", "sr", "split", "subset",
                      "text", "phones", "gender", "style", "noise", "device", "dialect", "offset", "consent_id",
                      "role", "sha256"}          # sha256: content hash written by datasets/arabic_import.py
PARTICIPANT_SPEAKER = re.compile(r"^own_recordings:spk-[0-9a-f]{10}$")
PARTICIPANT_SESSION = re.compile(r"^ses-[0-9a-f]{8}$")
PARTICIPANT_CONSENT = re.compile(r"^CNS-[0-9a-f]{12}$")
_RANDOM_ID = re.compile(r"(?<![0-9A-Za-z])(?:spk-[0-9a-f]{10}|ses-[0-9a-f]{8}|rec-[0-9a-f]{16}|CNS-[0-9a-f]{12})"
                        r"(?![0-9A-Za-z])")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_Y = r"(?:19|20|14)\d{2}"                     # Gregorian 19xx/20xx, Hijri 14xx
_M = r"(?:0?[1-9]|1[0-2])"
_D = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?"
           r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_AR_MONTHS = ("(?:يناير|فبراير|مارس|أبريل|ابريل|إبريل|مايو|يونيو|يونيه|يوليو|يوليه|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر"
              "|نوفمبر|ديسمبر|كانون الثاني|كانون الأول|كانون الاول|شباط|آذار|اذار|نيسان|أيار|ايار|حزيران|تموز|آب"
              "|أيلول|ايلول|تشرين الأول|تشرين الاول|تشرين الثاني|محرم|صفر|ربيع الأول|ربيع الاول|ربيع الآخر"
              "|ربيع الثاني|جمادى الأولى|جمادى الاولى|جمادى الآخرة|جمادى الاخرة|رجب|شعبان|رمضان|شوال|ذو القعدة"
              "|ذو الحجة)")
_AR = "؀-ۿ"
FINER_THAN_MONTH = [
    ("date", re.compile(rf"(?<!\d){_Y}[-/._ ]{_M}[-/._ ]{_D}(?!\d)")),                   # 2026-10-11, 2026_10_1
    ("date", re.compile(r"(?<!\d)(?:19|20|14)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)")),  # 20261011
    ("date", re.compile(rf"(?<!\d){_D}[-/._ ]{_M}[-/._ ]{_Y}(?!\d)")),                   # 11-10-2026
    ("date", re.compile(r"(?<!\d)(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:19|20)\d{2}(?!\d)")),     # 11102026
    ("date", re.compile(rf"(?<!\d){_D}[-/]{_M}[-/]\d{{2}}(?!\d)")),                      # 11-10-26, 10/11/26
    ("date", re.compile(r"(?<!\d)(?:19|20)\d{2}-?(?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6])(?!\d)")),  # ordinal 2026-284
    ("date", re.compile(r"(?i)(?<!\d)(?:19|20)\d{2}-?w(?:0[1-9]|[1-4]\d|5[0-3])(?![0-9])")),        # ISO week 2026-W41
    ("date", re.compile(r"(?i)(?<![a-z])" + _MONTHS + r"[-_. ]?" + _D + r"(?!\d)")),     # Oct11, october 11
    ("date", re.compile(r"(?i)(?<!\d)" + _D + r"[-_. ]?" + _MONTHS + r"(?![a-z])")),     # 11oct2026
    ("date", re.compile(rf"(?<![{_AR}])" + _AR_MONTHS + rf"\s*[-_/]?\s*{_D}(?!\d)")),    # أكتوبر 11
    ("date", re.compile(rf"(?<!\d){_D}\s*[-_/]?\s*" + _AR_MONTHS + rf"(?![{_AR}])")),    # 11 تشرين الأول
    ("time", re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)")),                    # 14:03
    ("time", re.compile(r"(?<![0-9A-Za-z])(?:[01]\d|2[0-3])h[0-5]\d(?![0-9A-Za-z])")),    # 14h03 (not inside ids)
    ("time", re.compile(r"(?<!\d)(?:[01]\d|2[0-3])[-_.][0-5]\d[-_.][0-5]\d(?!\d)")),     # 14-03-22
    ("epoch", re.compile(r"(?<![0-9A-Za-z])1\d{9}(?:\d{3})?(?![0-9A-Za-z])")),           # 1760191402(000)
]


def _ascii_digits(value):
    """Map every Unicode decimal digit (Arabic-Indic ٠-٩, Eastern Arabic-Indic ۰-۹, ...) to ASCII."""
    return "".join(str(unicodedata.decimal(ch)) if not ch.isascii() and unicodedata.decimal(ch, None) is not None
                   else ch for ch in str(value))


def finer_than_month(value):
    """-> the kind of the first date/time finer than a month found in `value` ("date", "time",
    "epoch"), or None. Month precision ("2026-10", "أكتوبر 2026") is allowed by the policy."""
    v = _ascii_digits(value)
    for kind, rx in FINER_THAN_MONTH:
        if rx.search(v):
            return kind
    return None


# ------------------------------------------------------------------ C9: personal-data indicators in text
# Separate from the timing check. Transcripts are speech content, so `text` is not rewritten and
# not covered by the timing rule; it is screened for INDICATORS of personal data: contact data
# (datasets/consent.CONTACT_PATTERNS: e-mail, phone/id-like numbers, @handles, t.me / tg://
# links, "id: 123..."), URLs, dates/times finer than a month, self-introductions ("اسمي ...",
# "my name is ..."), titles before a name ("الأستاذ ...", "Dr ..."), kunyas ("أبو ..."), and
# address words next to a number ("شارع ... 12", "building 4"). Only the KINDS are reported,
# never the text. A hit means "needs human review"; no hit does NOT mean the text is free of
# personal data (names alone, places, spelled-out numbers and free wording are not detected).
_TEXT_EXTRA = [
    ("url", re.compile(r"https?://|(?<![\w@])www\.|(?<![\w@])[a-z0-9-]+\.(?:com|net|org|io|jo|sa|ae|eg|me|ly)\b")),
    ("self_introduction", re.compile(r"(?:اسمي|إسمي|أدعى|ادعى|my name is|my name's|i am called|call me)\s+\S+")),
    ("title_with_name", re.compile(r"(?<!\S)(?:السيد|السيدة|الآنسة|الانسة|الأستاذ|الاستاذ|الأستاذة|الاستاذة|الدكتور"
                                   r"|الدكتورة|المهندس|المهندسة|الحاج|الحاجة|mr|mrs|ms|miss|dr)\.?\s+\S+")),
    ("kunya", re.compile(r"(?<!\S)(?:أبو|ابو|أبي|ابي)\s+\S+")),
    ("address", re.compile(r"(?<!\S)(?:شارع|حارة|حي|عمارة|بناية|مبنى|شقة|طابق|دوار|street|st\.|road|rd\.|avenue|ave\."
                           r"|building|apt|apartment|floor|p\.?\s?o\.?\s?box|ص\.?\s?ب)\s*\S{0,20}\s*\d+")),
]
_CONTACT_KINDS = ("email", "phone_or_id_number", "handle", "telegram_link", "telegram_link", "id_number")


def text_privacy_indicators(text):
    """-> sorted list of personal-data indicator KINDS found in `text` (empty = none found, which
    is not a guarantee). Never returns the matched text."""
    from datasets.consent import CONTACT_PATTERNS, normalise_text
    t = normalise_text(text)                      # NFKC, invisible chars removed, digits -> ASCII, lower case
    kinds = {k for k, rx in zip(_CONTACT_KINDS, CONTACT_PATTERNS) if rx.search(t)}
    kinds |= {k for k, rx in _TEXT_EXTRA if rx.search(t)}
    if finer_than_month(t):
        kinds.add("date_or_time")
    return sorted(kinds)


def participant_text_checks(rows):
    """Participant rows whose text shows personal-data indicators are refused (human review);
    messages carry the row and the kinds only."""
    errors = []
    for i, r in enumerate(rows):
        if _is_participant(r) and r.get("text"):
            kinds = text_privacy_indicators(r["text"])
            if kinds:
                errors.append(f"row {i}: participant text has personal-data indicators {kinds} (needs human review)")
    return errors


def _is_participant(r):
    return r.get("corpus") == "own_recordings" or "consent_id" in r


def participant_timing_checks(rows, inputs=None):
    """C9: refuse participant rows that could carry timing finer than a month. Error messages
    name the row, field and kind only, never the offending value."""
    errors = []
    participant = False
    for i, r in enumerate(rows):
        if not _is_participant(r):
            continue
        participant = True
        extra = sorted(set(r) - PARTICIPANT_FIELDS)
        if extra:
            errors.append(f"row {i}: participant row has field(s) {extra} outside the manifest format "
                          "(dates/times must not enter manifests)")
        for k, rx, form in (("speaker", PARTICIPANT_SPEAKER, "own_recordings:spk-<10 hex>"),
                            ("session", PARTICIPANT_SESSION, "ses-<8 hex>"),
                            ("consent_id", PARTICIPANT_CONSENT, "CNS-<12 hex>")):
            if not rx.match(str(r.get(k, ""))):
                errors.append(f"row {i}: participant {k} is not a random schema id ({form})")
        if "sha256" in r and not _SHA256.match(str(r["sha256"])):
            errors.append(f"row {i}: participant field 'sha256' is not a sha256 hex digest")
        for k in sorted(set(r) & PARTICIPANT_FIELDS - {"text", "speaker", "session", "consent_id", "duration", "sr",
                                                        "offset", "sha256"}):
            kind = finer_than_month(_RANDOM_ID.sub("<id>", str(r[k])))
            if kind:
                errors.append(f"row {i}: participant field {k!r} contains a {kind} finer than a month")
    if participant and inputs is not None:
        kind = finer_than_month(json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False))
        if kind:
            errors.append(f"manifest_index inputs contain a {kind} finer than a month (participant data present)")
    return errors


def assert_no_leakage(rows):
    errs = (validate(rows) + leakage_check(rows) + session_checks(rows) + participant_timing_checks(rows)
            + participant_text_checks(rows))
    if errs:
        raise LeakageError("; ".join(errs[:20]))


def licence_check(rows, path, consents=None, prov=None):
    """Strict dataset gate (datasets/gate.py): commercial path admits only COMMERCIAL_VERIFIED
    corpora and consent-eligible own recordings; research path also COMMERCIAL_PENDING and
    RESEARCH_ONLY. LICENSE_UNVERIFIED / unknown corpora are NOT_ELIGIBLE on every path.
    own_recordings rows always need per-speaker consent records (datasets/consent.py)."""
    from datasets.gate import check_rows as gate_rows
    errs = set(gate_rows([r for r in rows if r["corpus"] != "own_recordings"], path, prov))
    own = [r for r in rows if r["corpus"] == "own_recordings"]
    if own and consents is None:
        errs.add("own_recordings rows need --consent-store (per-speaker consent records)")
    elif own:
        from datasets.consent import check_rows
        errs.update(check_rows(own, consents, path))
    return sorted(errs)


def stats(rows):
    agg = defaultdict(lambda: {"utterances": 0, "hours": 0.0, "speakers": set(), "sessions": set()})
    for r in rows:
        for key in ((r["split"], "ALL"), (r["split"], r["corpus"]), (r["split"], r["language"])):
            a = agg[key]
            a["utterances"] += 1
            a["hours"] += r["duration"] / 3600
            a["speakers"].add(r["speaker"])
            a["sessions"].add((r["speaker"], r["session"]))
    return {f"{s}/{k}": {"utterances": v["utterances"], "hours": round(v["hours"], 3),
                         "speakers": len(v["speakers"]), "sessions": len(v["sessions"])}
            for (s, k), v in sorted(agg.items())}


def write(rows, out_dir, seed, inputs, dropped):
    assert_no_leakage(rows)                 # every writer is gated, not only the callers that remember to check
    errs = participant_timing_checks(rows, inputs)
    if errs:
        raise LeakageError("; ".join(errs[:20]))
    os.makedirs(out_dir, exist_ok=True)
    index = {"seed": seed, "inputs": inputs, "dropped": dropped, "manifests": {}}
    for split in ("train", "valid", "test", "attacker_train"):
        p = os.path.join(out_dir, f"{split}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        index["manifests"][split] = _sha(p)
    st = stats(rows)
    json.dump(index, open(os.path.join(out_dir, "manifest_index.json"), "w"), indent=1)
    json.dump(st, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "stats.md"), "w") as f:
        f.write("| split/group | utterances | hours | speakers | sessions |\n|---|---|---|---|---|\n")
        for k, v in st.items():
            f.write(f"| {k} | {v['utterances']} | {v['hours']} | {v['speakers']} | {v['sessions']} |\n")
    return index


def verify(out_dir):
    """Re-check written manifests before training: every sha256 in manifest_index.json must
    match the file on disk and the union must pass the leakage/session checks. Raises
    LeakageError on any mismatch or finding (trainers call this in preflight)."""
    with open(os.path.join(out_dir, "manifest_index.json")) as f:
        index = json.load(f)
    rows = []
    for split, digest in index["manifests"].items():
        p = os.path.join(out_dir, f"{split}.jsonl")
        if _sha(p) != digest:
            raise LeakageError(f"{p}: sha256 differs from manifest_index.json (edited after build)")
        rows += _jsonl(p)
    assert_no_leakage(rows)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", help="verify an existing manifest directory instead of building")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--librispeech", action="append", default=[], help="ROOT:CORPUS:SUBSET[,SUBSET]")
    ap.add_argument("--vctk", action="append", default=[])
    ap.add_argument("--commonvoice", action="append", default=[], help="ROOT:LANG")
    ap.add_argument("--segments", action="append", default=[], help="TSV:CORPUS[:LANG]")
    ap.add_argument("--generic", action="append", default=[], help="ROOT:CORPUS:LANG")
    ap.add_argument("--consent-store", help="private JSONL of consent records for own_recordings (data/consent_schema.json)")
    ap.add_argument("--licence-path", choices=["commercial", "research"], default="commercial",
                    help="research-path models must never ship (docs/DATASET_LICENSE_MATRIX.md)")
    a = ap.parse_args(argv)
    if a.verify:
        try:
            verify(a.verify)
        except LeakageError as e:
            print("LEAKAGE: " + str(e), file=sys.stderr)
            sys.exit(1)
        print("manifests verified: hashes match, no leakage")
        return
    if not a.out:
        ap.error("--out is required when building")
    rows = []
    for spec in a.librispeech:
        root, corpus, subs = spec.rsplit(":", 2)
        rows += build_librispeech(root, corpus, subs.split(","))
    for root in a.vctk:
        rows += build_vctk(root)
    for spec in a.commonvoice:
        root, lang = spec.rsplit(":", 1)
        rows += build_commonvoice(root, lang)
    for spec in a.segments:
        parts = spec.split(":")
        rows += build_segments(parts[0], parts[1], parts[2] if len(parts) > 2 else "en")
    for spec in a.generic:
        root, corpus, lang = spec.rsplit(":", 2)
        rows += build_generic(root, corpus, lang)
    split_rows, dropped = assign_splits(rows, a.seed)
    try:
        assert_no_leakage(split_rows)
    except LeakageError as e:
        print("LEAKAGE: " + str(e), file=sys.stderr)
        sys.exit(1)
    consents = None
    if a.consent_store:
        from datasets.consent import load_store
        consents = load_store(a.consent_store)
    lic = licence_check(split_rows, a.licence_path, consents)
    if lic:
        print("LICENCE: " + "; ".join(lic), file=sys.stderr)
        sys.exit(1)
    idx = write(split_rows, a.out, a.seed, dict(vars(a), licence_path=a.licence_path), dropped)
    print(json.dumps(idx["manifests"], indent=1))


if __name__ == "__main__":
    main()
