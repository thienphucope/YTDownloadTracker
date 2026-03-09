import modal
import torch
from typing import Dict, Any
from fastapi.responses import FileResponse
import os
import uuid

# 1. Khai báo App
app = modal.App("xtts-api")

# 2. Volume để lưu model và file âm thanh mẫu (reference audio)
# Lệnh tạo: modal volume create tts-local-cache
tts_cache = modal.Volume.from_name("tts-local-cache")

# 3. Build Image
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
    # Set biến môi trường để cache model vào Volume
    .env({"TTS_HOME": "/cache"})
)

# 4. Định nghĩa Class xử lý TTS
@app.cls(
    gpu="L4", 
    image=image,
    timeout=1800,
    max_containers=1,
    volumes={"/cache": tts_cache},
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class XTTSAPI:
    
    @modal.enter(snap=True)
    def load_model_to_gpu(self):
        """Chạy một lần khi tạo snapshot, load model thẳng vào VRAM"""
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        from TTS.api import TTS
        print("🚀 Loading XTTS model to GPU...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model tts_v2
        self.tts = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(device)
        
        # Đường dẫn file mẫu trong Volume
        self.speaker_wav_path = "/cache/refs/chunk_0004.wav"
        
        print(f"✅ Model loaded on {device} and snapshotted!")

    @modal.method()
    def generate(self, text: str, language: str = "en") -> str:
        """Hàm nội bộ để tạo file âm thanh"""
        if not text:
            raise ValueError("Text input is required")
        
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"/tmp/output_{unique_id}.wav"
        
        # Kiểm tra file ref có tồn tại không
        ref_path = self.speaker_wav_path if os.path.exists(self.speaker_wav_path) else None
        if not ref_path:
            print("⚠️ Warning: Reference audio not found. Using default voice.")

        self.tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=output_path
        )
        
        return output_path

    @modal.fastapi_endpoint(method="GET")
    def ping(self) -> Dict[str, Any]: # Đã sửa lỗi: dùng Any để nhận kiểu Boolean (True)
        return {
            "status": "ok", 
            "model_loaded": True, 
            "gpu": "L4",
            "info": "XTTS v2 is ready"
        }

    @modal.fastapi_endpoint(method="POST")
    def tts_generate(self, body: Dict[str, Any]) -> FileResponse:
        """Endpoint chính để nhận text và trả về file âm thanh"""
        text = body.get("text", "")
        language = body.get("language", "en")
        
        if not text:
            # FastAPI sẽ tự trả về 422 nếu bạn dùng Pydantic, 
            # nhưng ở đây ta check thủ công cho đơn giản.
            return {"error": "Missing 'text' in request body"}
        
        # Gọi hàm generate trên cùng container
        wav_path = self.generate.local(text=text, language=language)
        
        return FileResponse(
            path=wav_path,
            media_type="audio/wav",
            filename=f"output_{language}.wav"
        )

# Triển khai: modal deploy <tên_file>.py