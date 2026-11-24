# Kaggle batch kernel: held-out evaluation of the VCTK ECAPA encoder (see ../vctk_speaker_encoder_eval_cell.py).
# Push: kaggle kernels push -p training/configs/kaggle/kernel_vctk_eval   (CPU, no accelerator)
# Poll: kaggle kernels status y1x00r/voise-vctk-speaker-encoder-eval ; log: kaggle kernels logs ...
import subprocess
import urllib.request

subprocess.run("rm -rf /kaggle/working/Voise", shell=True, check=True)
URL = ("https://raw.githubusercontent.com/Y1X0/Voise/ccr-a40f5481-ilh4t8/"
       "training/configs/kaggle/vctk_speaker_encoder_eval_cell.py")
exec(urllib.request.urlopen(URL).read())
