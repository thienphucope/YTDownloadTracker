import modal
import torch
from typing import Dict, Any
from fastapi.responses import FileResponse, StreamingResponse
import io
import os
import uuid
import re
import sys
import time
import numpy as np

# 1. Khai báo App
app = modal.App("xtts-ft-api")

# 2. Volume
tts_cache = modal.Volume.from_name("tts-local-cache")

# 3. Cấu hình Repo và Thư mục
REPO_URL = "https://github.com/thienphucope/XTTSv2-Finetuning-for-New-Languages"
CODE_DIR = "/root/xtts-ft"
MODEL_DIR = "/cache/elinoir"

# 4. Build Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers==4.37.2", "accelerate==0.27.2", "scipy",
        "fastapi", "requests", "numpy<2.0", "pydantic", "omegaconf"
    )
    .pip_install("pysbd", "coqpit", "anyascii", "trainer", "soundfile", "librosa")
    .run_commands(f"rm -rf {CODE_DIR} && git clone {REPO_URL} {CODE_DIR}")
    .env({"PYTHONPATH": CODE_DIR})
)

# 5. Định nghĩa Class xử lý TTS
@app.cls(
    gpu="L4", 
    image=image,
    timeout=180,
    max_containers=1,
    volumes={"/cache": tts_cache},
    scaledown_window=180,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class XTTSFTAPI:
    
    @modal.enter(snap=True)
    def load_model_to_gpu(self):
        """Chạy một lần khi tạo snapshot, load model fine-tuned từ folder elinoir/"""
        if CODE_DIR not in sys.path:
            sys.path.append(CODE_DIR)
            
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        print("🚀 Loading Fine-tuned XTTS model to GPU...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        xtts_checkpoint = f"{MODEL_DIR}/checkpoint_686.pth"
        xtts_config = f"{MODEL_DIR}/config.json"
        xtts_vocab = f"{MODEL_DIR}/vocab.json"
        
        if not os.path.exists(xtts_config):
            raise FileNotFoundError(f"Missing required model file: {xtts_config}")

        config = XttsConfig()
        config.load_json(xtts_config)
        
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, 
            checkpoint_path=xtts_checkpoint, 
            vocab_path=xtts_vocab, 
            use_deepspeed=False
        )
        self.model.to(device)
        
        # Đường dẫn file mẫu trong Volume
        self.speaker_wav_path = "/cache/refs/schrolee.wav"
        self.is_interrupted = False
        
        print(f"✅ Fine-tuned Model loaded on {device} and snapshotted!")

    def _get_conditioning_latents(self, ref_path: str):
        """Helper để lấy đặc trưng giọng nói"""
        if not ref_path or not os.path.exists(ref_path):
            print(f"⚠️ Warning: Reference audio {ref_path} not found.")
            return None, None
            
        return self.model.get_conditioning_latents(
            audio_path=[ref_path],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

    @modal.method()
    def generate(self, text: str, language: str = "en") -> str:
        """Hàm nội bộ để tạo file âm thanh sử dụng model fine-tuned"""
        if not text:
            raise ValueError("Text input is required")
        
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"/tmp/output_{unique_id}.wav"
        
        ref_path = self.speaker_wav_path if os.path.exists(self.speaker_wav_path) else None
        gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(ref_path)
        
        out = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
        )
        
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, 24000, out["wav"])
        
        return output_path

    @modal.fastapi_endpoint(method="GET")
    def ping(self) -> Dict[str, Any]:
        return {
            "status": "ok", 
            "model_loaded": True, 
            "gpu": "L4",
            "info": "XTTS v2 Fine-tuned is ready"
        }

    @modal.fastapi_endpoint(method="POST")
    def tts_interrupt(self) -> Dict[str, Any]:
        self.is_interrupted = True
        print("🛑 Nhận được cờ Interrupt!")
        return {"status": "interrupted"}

    @modal.fastapi_endpoint(method="POST")
    def tts_generate(self, body: Dict[str, Any]) -> FileResponse:
        text = body.get("text", "")
        language = body.get("language", "en")
        
        if not text:
            return {"error": "Missing 'text' in request body"}
        
        wav_path = self.generate.local(text=text, language=language)
        
        return FileResponse(
            path=wav_path,
            media_type="audio/wav",
            filename=f"output_{language}.wav"
        )
        
    @modal.fastapi_endpoint(method="POST")
    def tts_stream(self, body: Dict[str, Any]) -> StreamingResponse:
        text = body.get("text", "")
        language = body.get("language", "en")
        
        if not text:
            return {"error": "Missing 'text' in request body"}
            
        def generate_audio_stream():
            ref_path = self.speaker_wav_path if os.path.exists(self.speaker_wav_path) else None
            gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(ref_path)
            
            # Logic tách chuỗi
            def split_text_smart(text_input, min_words=10, max_words=25):
                text_input = re.sub(r'[*#_~-]', ' ', text_input)
                text_input = text_input.replace('\n', '. ')
                raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_input) if s.strip()]
                return raw_sentences if raw_sentences else [text_input]

            text_chunks = split_text_smart(text)
            self.is_interrupted = False
            
            for chunk_text in text_chunks:
                if self.is_interrupted: break
                if not chunk_text.strip(): continue
                    
                chunk_for_xtts = re.sub(r'[.!?\n]', ',', chunk_text)
                
                audio_stream = self.model.inference_stream(
                    text=chunk_for_xtts,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.7
                )
                
                accumulated_chunks = []
                for chunk in audio_stream:
                    if self.is_interrupted: break
                    chunk_np = chunk.cpu().numpy()
                    accumulated_chunks.append(chunk_np)
                    if len(accumulated_chunks) >= 3:
                        combined = np.concatenate(accumulated_chunks)
                        pcm_data = (combined * 32767).astype(np.int16).tobytes()
                        yield pcm_data
                        accumulated_chunks = []
                if accumulated_chunks:
                    combined = np.concatenate(accumulated_chunks)
                    pcm_data = (combined * 32767).astype(np.int16).tobytes()
                    yield pcm_data
                
        return StreamingResponse(generate_audio_stream(), media_type="audio/pcm")
