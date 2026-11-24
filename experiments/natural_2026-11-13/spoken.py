import sys, os, json; sys.path.insert(0,"training/scripts")
import numpy as np, soundfile as sf, librosa, torch, voice_clone_tts as V
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
A="models/whisper-turbo-dialectal-v2-ct2"; out="runs/natural_omar/spoken"
T=["يعني بصراحة ما كنت متوقع إنه الموضوع يطلع هيك، بس خلص، صار اللي صار.",
   "اه والله، امبارح سهرنا لوقت متأخر، وهيك يعني ما نمت منيح.",
   "شوف، أنا رأيي إنه نستنى شوي، لأنه يعني لسا بكير نقرر.",
   "لا لا لا، مش هيك القصة، خليني أحكيلك من الأول.",
   "طب اسمع، بكرا إذا فاضي بنطلع نشرب قهوة ومنحكي."]
m=OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2",device_map="cpu")
m.load_state_dict(torch.load("runs/personal_omar_r1/omar_personal.pt",map_location="cpu",weights_only=True)["weights"],strict=False); m.eval()
ref="runs/personal_omar_r1/ref.wav"; rt=V.transcribe(sf.read(ref,dtype="float32")[0],A)
p=m.create_voice_clone_prompt(ref_audio=ref,ref_text=rt); res=[]
for i,t in enumerate(T):
    torch.manual_seed(0)
    with torch.no_grad():
        y=m.generate(text=t,language="apc",voice_clone_prompt=p,generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
    y=np.asarray(y[0] if isinstance(y,list) else y,np.float32).squeeze(); y=y/max(1e-6,float(np.abs(y).max()))*0.9
    w,k=V.watermark(y,24000); assert k=="perth"
    sf.write(f"{out}/SYNTHETIC_omar_spoken_{i+1}.wav",w,24000,subtype="PCM_16")
    h=V.transcribe(librosa.resample(y,orig_sr=24000,target_sr=16000),A)
    res.append(dict(text=t,seconds=round(len(y)/24000,2),cer=round(V.cer(t,h),3),heard=h)); print("S",res[-1],flush=True)
json.dump(res,open(f"{out}/results.json","w"),ensure_ascii=False,indent=1); print("DONE")
