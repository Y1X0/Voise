# Kaggle batch kernel: runs the VCTK speaker-encoder cell from the pushed branch (see ../vctk_speaker_encoder_cell.py).
# Push: kaggle kernels push -p training/configs/kaggle/kernel_vctk --accelerator NvidiaTeslaT4
# Poll: kaggle kernels status y1x00r/voise-vctk-speaker-encoder ; logs/outputs: kaggle kernels output ... -p <dir>
import subprocess
import urllib.request

subprocess.run("rm -rf /kaggle/working/Voise", shell=True, check=True)
URL = ("https://raw.githubusercontent.com/Y1X0/Voise/claude/voice-anonymizer-android-az4mcs/"
       "training/configs/kaggle/vctk_speaker_encoder_cell.py")
exec(urllib.request.urlopen(URL).read())
