import os
import warnings
# Khóa log từ hệ thống
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

import torch
import torchaudio
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Tối ưu cho GPU RTX của HP Victus
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1. Load pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="HF_TOKEN"
)
pipeline.to(torch.device("cuda"))

# 2. Load audio
audio_path = r"E:\speechGen\downloads\solus\wavs\e0h3O3qqLPs.wav"
data, samplerate = sf.read(audio_path)
waveform = torch.tensor(data).float().t()
if len(waveform.shape) == 1:
    waveform = waveform.unsqueeze(0)

audio_data = {"waveform": waveform, "sample_rate": samplerate}

# 3. Chạy
print(f"--- Đang xử lý Diarization trên GPU ---")
with ProgressHook() as hook:
    output = pipeline(audio_data, hook=hook)

# 4. In kết quả gọn gàng
print("\n--- KẾT QUẢ PHÂN ĐOẠN ---")
for turn, speaker in output.speaker_diarization:
    print(f"[{turn.start:7.1f}s - {turn.end:7.1f}s] -> {speaker}")