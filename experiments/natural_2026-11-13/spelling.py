"""Do diacritics or another spelling fix a word the model pronounces wrong? Same voice, seed and settings as
spoken.py; only the spelling of one word changes. Judged by ear (Whisper ignores a shadda)."""
import sys, os, json; sys.path.insert(0, "training/scripts")
import numpy as np, soundfile as sf, torch, voice_clone_tts as V
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
out = "runs/natural_omar/lex"
CASES = [("bakkir", "اه بدي أصحى {} عشان ألحق الباص.", ["بكير", "بكّير", "بَكّير", "بكّيير"]),
         ("bukra", "خلص {} منحكي إن شاء الله.", ["بكرا", "بُكرا", "بكره", "بُكْرَة"])]
m = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
m.load_state_dict(torch.load("runs/personal_omar_r1/omar_personal.pt", map_location="cpu", weights_only=True)["weights"], strict=False)
m.eval()
ref = "runs/personal_omar_r1/ref.wav"; rt = V.transcribe(sf.read(ref, dtype="float32")[0], "models/whisper-turbo-dialectal-v2-ct2")
p = m.create_voice_clone_prompt(ref_audio=ref, ref_text=rt)
for tag, frame, spellings in CASES:
    for j, s in enumerate(spellings):
        torch.manual_seed(0)
        with torch.no_grad():
            y = m.generate(text=frame.format(s), language="apc", voice_clone_prompt=p,
                           generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
        y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze(); y = y / max(1e-6, float(np.abs(y).max())) * 0.9
        w, k = V.watermark(y, 24000); assert k == "perth"
        sf.write(f"{out}/SYNTHETIC_omar_{tag}_{j + 1}.wav", w, 24000, subtype="PCM_16")
        print("S", tag, j + 1, frame.format(s), flush=True)
print("DONE")
