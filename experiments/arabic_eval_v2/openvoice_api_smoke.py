# Usage: python -I openvoice_api_smoke.py <OpenVoice@74a1d147> <Voise repo> <outdir containing converter config.json>
# API-mechanics smoke test only: random-init OpenVoice V2 converter (no checkpoint), synthetic signals, no real voice.
import sys, ast, json, numpy as np, soundfile as sf, torch
OV, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, OV)
from openvoice import utils, models, mel_processing     # NOT openvoice.api / openvoice.text / se_extractor
assert not any(m.startswith(("openvoice.api", "openvoice.text", "openvoice.se_extractor", "unidecode",
               "whisper_timestamped", "wavmark", "faster_whisper")) for m in sys.modules)
hps = utils.get_hparams_from_file(f"{OUT}/config.json")
net = models.SynthesizerTrn(0, hps.data.filter_length // 2 + 1, n_speakers=hps.data.n_speakers, **hps.model).eval()
# synthetic "pool": harmonic signals with vibrato (no speech, no person)
rng = np.random.default_rng(0); fs = 16000; paths = []
for i in range(6):
    t = np.arange(int(fs * 2.0)) / fs; f0 = 110 + 30 * i + 5 * np.sin(2 * np.pi * 4 * t)
    ph = 2 * np.pi * np.cumsum(f0) / fs
    x = sum(np.sin(k * ph) / k for k in range(1, 20)) * 0.1 + 0.002 * rng.standard_normal(t.size)
    p = f"{OUT}/pool{i}.wav"; sf.write(p, x.astype(np.float32), fs); paths.append(p)
src = open(f"{REPO}/scripts/neural_anonymization_eval.py").read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "PseudoSpeakerWorld")
ns = dict(np=np, sf=sf, sys=sys); exec(ast.get_source_segment(src, cls), ns)
psn = ns["PseudoSpeakerWorld"](paths, fs=16000, cmvn=True)
renders = []
for i, p in enumerate(paths):
    o = f"{OUT}/render{i}.wav"; psn.run(p, o, 4242, 0); renders.append(o)
import librosa
def spec(path):
    a, _ = librosa.load(path, sr=hps.data.sampling_rate)
    y = torch.FloatTensor(a).unsqueeze(0)
    return mel_processing.spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                                            hps.data.hop_length, hps.data.win_length, center=False)
def se(path_list):  # same arithmetic as ToneColorConverter.extract_se
    with torch.no_grad():
        return torch.stack([net.ref_enc(spec(p).transpose(1, 2)).unsqueeze(-1) for p in path_list]).mean(0)
tgt = se(renders); src_se = se([paths[0]])
s = spec(paths[0])
with torch.no_grad():
    y = net.voice_conversion(s, torch.LongTensor([s.size(-1)]), sid_src=src_se, sid_tgt=tgt, tau=0.3)[0][0, 0].numpy()
print(json.dumps(dict(tgt_se_shape=list(tgt.shape), out_samples=int(y.size), in_samples_22k=int(librosa.get_duration(path=paths[0]) * 22050),
                      finite=bool(np.isfinite(y).all()), forbidden_modules_loaded=False)))
