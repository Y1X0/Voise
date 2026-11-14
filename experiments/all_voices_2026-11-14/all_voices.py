"""One sentence in every voice of the learning pool, with the current learned version: the tool is for everyone in
the pool, not for one or two people. Each voice's consent record is read from its clips' provenance, and every output
is SYNTHETIC_*, watermarked, with a sidecar. Reports letter error, vowel mistakes and similarity per voice.

Usage: python all_voices.py OUT_DIR "sentence"
"""
import glob, json, os, sys
sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402

out, text = sys.argv[1], sys.argv[2]
eng = V.Engine("models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2", adapter="runs/learn/adapter.pt",
               sv_dir="models/ecapa-voxceleb", mos_path="models/utmos/utmos22_strong_step7459_v1.pt")
rows = []
for d in sorted(glob.glob("runs/learn/pool/*")):
    person = os.path.basename(d)
    cons = sorted({json.load(open(j)).get("consent_record") for j in glob.glob(d + "/*.json")})
    wav, summ, _ = eng.say(cons[-1], d, text, out, candidates=3, who=person)
    m = json.load(open(wav[:-4] + ".json", encoding="utf-8"))
    best = [c for p in m["pieces"] for c in p["candidates"] if c["seed"] == p["chosen_seed"]]
    rows.append(dict(person=person, wav=os.path.basename(wav), cer=summ["cer"], heard=summ["heard"],
                     vowels_wrong=sum(len(c.get("vowels_wrong") or []) for c in best),
                     sim=best[0].get("sim"), mos=summ["mos"]))
    print("V", rows[-1], flush=True)
json.dump(rows, open(os.path.join(out, "all_voices.json"), "w"), ensure_ascii=False, indent=1)
print("DONE")
