import modal
import os
import sys
import time
import io
import numpy as np
from typing import Dict, Optional
from fastapi import Response, HTTPException

# Upload Command: modal volume put chatterbox-models "wavs\scp461 đọc diễn cảm.wav" /checkpoints/

# --- CẤU HÌNH ---
APP_NAME = "viterbox-api"
REPO_URL = "https://github.com/thienphucope/viterbox-tts"
CODE_DIR = "/root/viterbox-tts"
MODEL_DIR = "/data/checkpoints" 

# Tên file ref bạn vừa upload (Nằm trong folder checkpoints)
REF_FILENAME = "scp513 cao giọng.wav" 

models_volume = modal.Volume.from_name("chatterbox-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    # PyTorch Stable
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("numpy<2.0")
    .pip_install(
        "transformers==4.44.2", "peft==0.13.2", "diffusers==0.29.0", "accelerate>=0.27.0",
        "resemble-perth==1.0.1", "conformer==0.3.2", "safetensors==0.5.3",
        "pykakasi==2.3.0", "librosa==0.11.0", "s3tokenizer", "sounddevice==0.5.2",
        "scipy", "fastapi", "uvicorn", "python-multipart", "omegaconf"
    )
    .run_commands(f"rm -rf {CODE_DIR} && git clone {REPO_URL} {CODE_DIR} && echo 'Timestamp: {int(time.time())}'")
    .env({"PYTHONPATH": CODE_DIR}) # CÁI NÀY DÙNG ĐỂ ĐỔI MODEL TỪ REPO GITHUB
)

app = modal.App(APP_NAME)

@app.cls(
    gpu="T4", 
    image=image,
    volumes={"/data": models_volume}, 
    scaledown_window=300,  
    timeout=600,
    enable_memory_snapshot=True 
)
class ViterboxApi:
    
    model_on_gpu = False

    @modal.enter(snap=True)
    def load_model_to_cpu(self):
        print("📥 Snapshot: Loading model to CPU RAM...")
        
        if CODE_DIR not in sys.path:
            sys.path.append(CODE_DIR)
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)

        try:
            import torch
            from viterbox import Viterbox
            
            # --- Monkey Patching để load vào CPU ---
            original_load = torch.load
            def cpu_load(*args, **kwargs):
                kwargs['map_location'] = torch.device('cpu')
                return original_load(*args, **kwargs)
            
            torch.load = cpu_load
            
            print(f"⏳ Loading Viterbox...")
            self.tts_engine = Viterbox.from_local(MODEL_DIR, device="cpu")
            
            torch.load = original_load # Trả lại nguyên trạng
            
            print("✅ Model loaded to CPU Memory!")
            self.model_on_gpu = False
            
        except Exception as e:
            print(f"❌ Snapshot Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def move_to_gpu_if_needed(self):
        if self.model_on_gpu:
            return

        print("🚀 Runtime: Moving model to GPU...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu": return

        t0 = time.time()
        try:
            # Đẩy model sang GPU
            if hasattr(self.tts_engine, "to"):
                self.tts_engine.to(device)
            else:
                for attr_name in dir(self.tts_engine):
                    attr = getattr(self.tts_engine, attr_name)
                    if isinstance(attr, torch.nn.Module) or isinstance(attr, torch.Tensor):
                        setattr(self.tts_engine, attr_name, attr.to(device))
            
            if hasattr(self.tts_engine, "device"):
                self.tts_engine.device = device
                
            print(f"✅ Moved to GPU in {time.time() - t0:.2f}s")
            self.model_on_gpu = True
        except Exception as e:
            print(f"❌ Failed to move to GPU: {e}")

    @modal.method()
    def generate_internal(self, text: str, config: Dict) -> bytes:
        t0 = time.time()
        import torch 
        import scipy.io.wavfile

        if CODE_DIR not in sys.path: sys.path.append(CODE_DIR)
        self.move_to_gpu_if_needed()
            
        print(f"🎤 Text: '{text[:50]}...'")
        
        try:
            lang = config.get("language", "vi")
            temp = config.get("temperature", 0.2)
            
            # --- CẤU HÌNH REF AUDIO ---
            # Đường dẫn file trong Volume
            ref_path = os.path.join(MODEL_DIR, REF_FILENAME)
            
            # Kiểm tra file có tồn tại không
            audio_prompt = None
            if os.path.exists(ref_path):
                # print(f"Sound found: {ref_path}")
                audio_prompt = ref_path
            else:
                print(f"⚠️ Warning: Không tìm thấy file ref tại {ref_path}. Dùng giọng mặc định.")

            # Gọi hàm generate
            audio_output = self.tts_engine.generate(
                text=text,
                language=lang,
                audio_prompt=audio_prompt, # <--- Đã truyền file ref vào đây
                exaggeration=config.get("exaggeration", 1.5),
                cfg_weight=config.get("cfg_weight", 0.05),
                temperature=temp,
                sentence_pause_ms=config.get("sentence_pause_ms", 0.5)
            )
            
            # --- Xử lý Output ---
            sample_rate = 24000
            audio_data = audio_output
            
            if isinstance(audio_output, tuple):
                sample_rate, audio_data = audio_output
            
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().cpu().numpy()
            
            audio_data = np.squeeze(audio_data)
            if audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1]: 
                audio_data = audio_data.T

            if audio_data.dtype.kind == 'f':
                 audio_data = np.clip(audio_data, -1.0, 1.0)
                 audio_data = (audio_data * 32767).astype(np.int16)
            
            buffer = io.BytesIO()
            scipy.io.wavfile.write(buffer, sample_rate, audio_data)
            buffer.seek(0)
            
            print(f"✅ Gen Time: {time.time() - t0:.2f}s")
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Generate Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

    @modal.fastapi_endpoint(method="POST")
    def tts(self, body: Dict) -> Response:
        text = body.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="Missing text")
        try:
            wav_bytes = self.generate_internal.local(text, body)
            return Response(content=wav_bytes, media_type="audio/wav")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok"}