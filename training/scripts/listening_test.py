"""Blind listening test (MUSHRA-lite) for comparing voice versions, and scoring of the answers.

  make   build a self-contained test folder: for every trial a real recording of the person (the anchor) and the
         systems' clips under random labels in a random order; listeners rate each clip 1-5 for "sounds like the
         person" and 1-5 for "sounds natural", then download their answers (a JSON file).
  score  combine answer files with the hidden key: mean and 95% confidence interval per system and question,
         and how often each system beat each other one on the same trial.

The folder holds consented voices, so it is for listening on this computer (open index.html); do not upload it.

  python listening_test.py make  --spec spec.json --out runs/listening/test1
  python listening_test.py score --test runs/listening/test1 --answers runs/listening/test1/answers/*.json

spec.json: {"title": "...", "trials": [{"person": "khaled", "real": "path.wav", "text": "...",
                                         "systems": {"original": "a.wav", "learned": "b.wav"}}, ...]}
"""
import argparse
import glob
import html
import itertools
import json
import os
import random
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_clone_tts as V  # noqa: E402

PAGE = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>اختبار استماع</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5d6677;--line:#dfe3ea;--accent:#2f6fde}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e8ebf1;--muted:#9aa3b4;--line:#2c3340;--accent:#6c9bff}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Segoe UI",Tahoma,system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 60px}.card{background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:16px;margin:0 0 14px}.muted{color:var(--muted);font-size:14px}audio{width:100%}
.clip{border-top:1px solid var(--line);padding:10px 0}.q{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:4px 0}
.q span{min-width:150px}label.o{border:1px solid var(--line);border-radius:8px;padding:2px 10px;cursor:pointer}
input[type=radio]{margin:0 4px 0 0}button{background:var(--accent);color:#fff;border:0;border-radius:10px;
padding:10px 20px;font:inherit;font-weight:700;cursor:pointer}input[type=text]{font:inherit;padding:6px 10px;border-radius:8px;
border:1px solid var(--line);background:var(--bg);color:var(--ink)}</style></head><body><main>
<h2>__TITLE__</h2>
<div class="card"><p>بكل سؤال في <b>تسجيل حقيقي</b> للشخص، وتحته كم <b>صوت مصنوع</b> بترتيب عشوائي.
اسمع الحقيقي أول، وبعدين قيّم كل صوت مصنوع:</p><ul><li><b>بيشبه الشخص؟</b> 1 = أبداً، 5 = هو بالزبط</li>
<li><b>طبيعي؟</b> 1 = آلي/مخربط، 5 = زي حكي إنسان حقيقي</li></ul>
<p class="muted">ما في جواب صح أو غلط. قيّم حسب أذنك. بالآخر اكبس «نزّل إجاباتي» وابعت الملف.</p>
<p>اسمك (أو أي رمز): <input type="text" id="rater" maxlength="40"></p></div>
<div id="trials"></div><button id="save">نزّل إجاباتي</button> <span class="muted" id="left"></span>
</main><script>
const T = __TRIALS__;
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const box = document.getElementById("trials");
const scale = (name) => [1,2,3,4,5].map(v => `<label class="o"><input type="radio" name="${name}" value="${v}">${v}</label>`).join("");
T.forEach((t, i) => {
  box.insertAdjacentHTML("beforeend", `<div class="card"><b>${i + 1}. </b><span class="muted">${esc(t.text)}</span>
    <div class="clip"><b>الحقيقي</b><audio controls preload="none" src="${esc(t.real)}"></audio></div>` +
    t.clips.map(c => `<div class="clip"><b>صوت ${esc(c.label)}</b><audio controls preload="none" src="${esc(c.file)}"></audio>
      <div class="q"><span>بيشبه الشخص؟</span>${scale(`s_${i}_${c.label}`)}</div>
      <div class="q"><span>طبيعي؟</span>${scale(`n_${i}_${c.label}`)}</div></div>`).join("") + `</div>`);
});
function answers() {
  const out = [];
  T.forEach((t, i) => t.clips.forEach(c => {
    const s = document.querySelector(`input[name="s_${i}_${c.label}"]:checked`);
    const n = document.querySelector(`input[name="n_${i}_${c.label}"]:checked`);
    out.push({trial: i, label: c.label, similar: s ? +s.value : null, natural: n ? +n.value : null});
  }));
  return out;
}
document.addEventListener("change", () => {
  const a = answers(); const missing = a.filter(x => !x.similar || !x.natural).length;
  document.getElementById("left").textContent = missing ? `ضايل ${missing}` : "كلهم مقيّمين ✓";
});
document.getElementById("save").onclick = () => {
  const blob = new Blob([JSON.stringify({test: "__ID__", rater: document.getElementById("rater").value, answers: answers()}, null, 1)],
                        {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `answers_${(document.getElementById("rater").value || "rater").replace(/[^\\w-]/g, "_")}.json`; a.click();
};
</script></body></html>"""


def make(a):
    spec = json.load(open(a.spec, encoding="utf-8"))
    out = V.check_out_dir(a.out)
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    os.makedirs(os.path.join(out, "answers"), exist_ok=True)
    rnd = random.Random(a.seed)
    trials, key = [], {}
    for i, t in enumerate(spec["trials"]):
        real = f"audio/t{i}_real{os.path.splitext(t['real'])[1]}"
        shutil.copy(t["real"], os.path.join(out, real))
        systems = list(t["systems"].items())
        rnd.shuffle(systems)
        clips = []
        for j, (name, path) in enumerate(systems):
            label = "ABCDEFGH"[j]
            f = f"audio/t{i}_{label}{os.path.splitext(path)[1]}"
            shutil.copy(path, os.path.join(out, f))
            clips.append({"label": label, "file": f})
            key[f"{i}:{label}"] = name
        trials.append({"text": t.get("text", ""), "real": real, "clips": clips, "person": t.get("person", "")})
    test_id = os.path.basename(os.path.normpath(out))
    page = (PAGE.replace("__TITLE__", html.escape(spec.get("title", "اختبار استماع")))
            .replace("__ID__", test_id)
            .replace("__TRIALS__", json.dumps([{k: v for k, v in t.items() if k != "person"} for t in trials],
                                              ensure_ascii=False)))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(page)
    json.dump({"test": test_id, "key": key, "trials": trials}, open(os.path.join(out, "key.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"{len(trials)} trials -> {out}/index.html (the hidden key is key.json; do not show it to listeners)")


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else None, None
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def score(a):
    key = json.load(open(os.path.join(a.test, "key.json"), encoding="utf-8"))["key"]
    files = [f for pat in a.answers for f in glob.glob(pat)]
    by = {}                                   # (system, question) -> ratings
    per_trial = {}                            # (rater, trial) -> {system: similar}
    raters = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        raters += 1
        for x in d["answers"]:
            name = key.get(f"{x['trial']}:{x['label']}")
            if name is None:
                continue
            for q in ("similar", "natural"):
                if x.get(q):
                    by.setdefault((name, q), []).append(x[q])
            if x.get("similar"):
                per_trial.setdefault((f, x["trial"]), {})[name] = x["similar"]
    systems = sorted({s for s, _ in by})
    res = {"raters": raters, "systems": {}}
    for s in systems:
        res["systems"][s] = {}
        for q in ("similar", "natural"):
            m, h = ci95(by.get((s, q), []))
            res["systems"][s][q] = {"mean": None if m is None else round(m, 2),
                                    "ci95": None if h is None else round(h, 2), "n": len(by.get((s, q), []))}
    wins = {}
    for s1, s2 in itertools.permutations(systems, 2):
        both = [v for v in per_trial.values() if s1 in v and s2 in v]
        if both:
            wins[f"{s1} > {s2}"] = round(sum(v[s1] > v[s2] for v in both) / len(both), 2)
    res["similar_wins"] = wins
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(a.test, "scores.json"), "w"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["make", "score"])
    ap.add_argument("--spec"); ap.add_argument("--out"); ap.add_argument("--test")
    ap.add_argument("--answers", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    (make if a.command == "make" else score)(a)


if __name__ == "__main__":
    main()
