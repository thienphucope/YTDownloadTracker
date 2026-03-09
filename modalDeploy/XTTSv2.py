import modal
import torch
from typing import Dict
from fastapi.responses import FileResponse
import os
import uuid  # Để tạo unique filename

# Tạo app Modal
app = modal.App("xtts-api")

# Volume persistent cho TTS cache (tạo bằng: modal volume create tts-local-cache)
tts_cache = modal.Volume.from_name("tts-local-cache")

# Build image với TTS và deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "TTS==0.22.0",
        "torch==2.4.1",
        "torchaudio",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "scipy",
        "fastapi",
        "requests",
    )
    .run_commands("export TTS_HOME=/cache")
)

# Class để load và cache model XTTS
@app.cls(
    gpu="L4",  # Sử dụng GPU A100 40GB
    image=image,
    timeout=1800,
    max_containers=1,
    volumes={"/cache": tts_cache},
    scaledown_window=300,
    
    # === THAY ĐỔI: Bật GPU Snapshot ===
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True}, # Bật GPU snapshot (alpha)
)
class XTTSAPI:
    
    # Giai đoạn 1: Load model thẳng vào GPU (để snapshot)
    # Hook này chạy khi tạo snapshot, CÓ GPU
    @modal.enter(snap=True)
    def load_model_to_gpu(self):
        # Fix TOS prompt
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        # Lazy import TTS
        from TTS.api import TTS
        print("Loading XTTS model directly to GPU (for GPU snapshot)...")

        # GPU đã có sẵn, có thể load thẳng
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        self.tts = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(device) # <-- Tải thẳng vào GPU
        
        # Ref audio từ Volume
        self.speaker_wav_path = "/cache/refs/no1.wav"
        if not os.path.exists(self.speaker_wav_path):
            print(f"Warning: Ref audio not found at {self.speaker_wav_path}. Using default speaker.")
            self.speaker_wav_path = None
        
        print("Model loaded to GPU and snapshotted!")

    # (Không cần hook thứ 2)

    @modal.exit()  # Cleanup tùy chọn
    def cleanup(self):
        print("Cleaning up...")
        if hasattr(self, 'tts'):
            del self.tts

    @modal.method()  # Method để generate WAV từ text
    def generate(self, text: str, language: str = "en") -> str:
        if not text:
            raise ValueError("Text input is required")
        
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"/tmp/output_{unique_id}_{language}.wav"
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_wav_path,
            language=language,
            file_path=output_path
        )
        
        if not os.path.exists(output_path):
            raise RuntimeError("Failed to generate WAV file")
        
        print(f"Generated WAV: {output_path}")
        return output_path

    # Ping endpoint để warm-up
    @modal.fastapi_endpoint(method="GET")
    def ping(self) -> Dict[str, str]:
        return {"status": "ok", "model_loaded": True, "gpu": "T4"}

    # Web endpoint: POST /
    @modal.fastapi_endpoint(method="POST")
    def tts_generate(self, body: Dict) -> FileResponse:
        text = body.get("text", "")
        language = body.get("language", "en")
        
        if not text:
            raise ValueError("Missing 'text' in request body")
        
        wav_path = self.generate.local(text=text, language=language)
        
        return FileResponse(
            path=wav_path,
            media_type="audio/wav",
            filename=f"xtts_output_{language}.wav"
        )

# Nhớ deploy bằng `modal deploy <tên_file>.py` để snapshot được tạo.