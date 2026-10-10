"""S1 + S2 (secondary, post hoc). See PLAN.md.

S1: compare SVQ `transcript_truth` with `text` (prompt) for the 225 frozen clips and recompute the in-run ASR
    hypotheses' CER/WER against `transcript_truth` as a SECONDARY reference.
S2: document the transport-corrupted reference string in results.json and show the in-run metric used intact text.

Usage: python3 reference_audit.py <experiment dir>   (reads manifest.json, results.json, secondary/transcript_truth.json)
Writes secondary/reference_audit.json and secondary/text_correction.json. results.json is never modified.
"""
import collections, json, sys, unicodedata
from math import comb
import numpy as np

D = sys.argv[1]
man = json.load(open(f"{D}/manifest.json"))["manifest"]
R = json.load(open(f"{D}/results.json"))
TT = json.load(open(f"{D}/secondary/transcript_truth.json"))["rows"]
SYS, ASRS = ("orig", "llvc", "dsp_strong", "dsp_balanced"), ("whisper", "mms")
clips = {c["utt_id"]: c for c in R["per_clip"]}


def norm(s):  # identical to evaluate.py
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا").replace("ة", "ه").replace("ى", "ي")
    return " ".join("".join(ch for ch in s if ch.isalpha() or ch == " ").split())


def ed(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return d[len(h)]


# ---------------- S2 ----------------
bad = [(u, k) for u, c in clips.items() for k, v in c.items() if isinstance(v, str) and "�" in v]
corr = []
for u, k in bad:
    m = next(c for c in man if c["utt_id"] == u)
    ref = norm(m[k])
    recomputed = {f"{a}_{x}_{s}": round((ed(ref, norm(clips[u][f"{a}_hyp_{s}"])) if x == "cer" else ed(ref.split(), norm(clips[u][f"{a}_hyp_{s}"]).split()))
                                        / max(1, len(ref) if x == "cer" else len(ref.split())), 4)
                  for a in ASRS for x in ("cer", "wer") for s in SYS}
    stored = {k2: clips[u][k2] for k2 in recomputed}
    corr.append(dict(utt_id=u, field=k, corrupted_value_in_results_json=clips[u][k], correct_value_from_frozen_manifest=m[k],
                     cause="Kaggle log transport split a multi-byte UTF-8 character; the run computed metrics in memory from the intact manifest text",
                     stored_metrics=stored, recomputed_with_manifest_text=recomputed,
                     all_stored_metrics_match=all(abs(stored[x] - recomputed[x]) < 1e-4 for x in stored)))
json.dump(dict(status="SECONDARY correction note; results.json is unchanged", n_corrupted_fields=len(bad), corrections=corr),
          open(f"{D}/secondary/text_correction.json", "w"), ensure_ascii=False, indent=1)

# ---------------- S1 ----------------
loc_of = {c["utt_id"]: c["locale"] for c in man}
spk_of = {c["utt_id"]: c["speaker"] for c in man}
stat = collections.defaultdict(collections.Counter)
for c in man:
    r = TT[c["utt_id"]]; t, g = r["text"], r.get("transcript_truth")
    assert t == c["text"]
    for k in (c["locale"], "all"):
        s = stat[k]; s["n"] += 1; s["missing"] += g in (None, ""); s["flag_transcriptions_speech_true"] += bool(r.get("transcriptions/speech"))
        if g:
            s["exact_raw"] += g == t; s["exact_normalised"] += norm(g) == norm(t)
            s["char_edits"] += ed(norm(t), norm(g)); s["chars"] += len(norm(t))
            s["word_edits"] += ed(norm(t).split(), norm(g).split()); s["words"] += len(norm(t).split())
s1 = {k: dict(v, char_edit_rate=round(v["char_edits"] / max(1, v["chars"]), 4), word_edit_rate=round(v["word_edits"] / max(1, v["words"]), 4))
      for k, v in stat.items()}

# intelligibility vs transcript_truth (secondary reference)
per = {}
for u, c in clips.items():
    g = norm(TT[u]["transcript_truth"])
    for a in ASRS:
        for s in SYS:
            h = norm(c[f"{a}_hyp_{s}"])
            per[(u, a, s)] = dict(cer=ed(g, h) / max(1, len(g)), wedits=ed(g.split(), h.split()), words=len(g.split()))


def summary(us, a):
    out = {}
    cw = lambda s: sum(per[(u, a, s)]["wedits"] for u in us) / max(1, sum(per[(u, a, s)]["words"] for u in us))
    for s in SYS:
        out[s] = dict(cer_median=round(float(np.median([per[(u, a, s)]["cer"] for u in us])), 4), corpus_wer=round(cw(s), 4))
        if s != "orig":
            out[s]["wer_rel_increase"] = round((cw(s) - cw("orig")) / cw("orig"), 4)
    d = collections.defaultdict(list)
    for u in us:
        d[spk_of[u]].append(per[(u, a, "llvc")]["cer"] - per[(u, a, "dsp_strong")]["cer"])
    sm = np.array([np.median(v) for v in d.values()]); k, m = int((sm > 0).sum()), int((sm != 0).sum())
    out["llvc_minus_dsp_strong"] = dict(cer_median_diff=round(float(np.median([per[(u, a, "llvc")]["cer"] - per[(u, a, "dsp_strong")]["cer"] for u in us])), 4),
                                        speakers_llvc_worse=f"{k}/{m}",
                                        speaker_sign_test_p_two_sided=float(min(1.0, 2 * min(sum(comb(m, i) for i in range(k, m + 1)), sum(comb(m, i) for i in range(0, k + 1))) / 2 ** m)) if m else None)
    return out


us_all = list(clips)
intel = {a: dict(all=summary(us_all, a), **{l: summary([u for u in us_all if loc_of[u] == l], a) for l in sorted(set(loc_of.values()))}) for a in ASRS}
# primary (prompt-referenced) values for side-by-side comparison
prim = {a: {s: dict(cer_median=R["intelligibility"][a][s]["cer_median"], corpus_wer=R["intelligibility"][a][s]["corpus_wer"]) for s in SYS} for a in ASRS}
json.dump(dict(status="SECONDARY post hoc; reference = SVQ transcript_truth; primary metrics (reference = prompt text) unchanged",
               text_vs_transcript_truth=s1, intelligibility_vs_transcript_truth=intel, primary_prompt_reference_for_comparison=prim),
          open(f"{D}/secondary/reference_audit.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(s1=s1["all"], s2=[(c["utt_id"], c["all_stored_metrics_match"]) for c in corr]), ensure_ascii=False))
