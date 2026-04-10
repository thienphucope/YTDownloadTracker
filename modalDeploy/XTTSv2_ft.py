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
MODEL_DIR = "/cache/tts/elinoir"

# 4. Build Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers==4.38.2", "accelerate>=0.27.0", "scipy",
        "fastapi", "requests", "numpy<2.0", "pydantic", "omegaconf", "einops>=0.6.0"
    )
    .pip_install(
        # Core & XTTS deps từ requirements.txt
        "mutagen==1.47.0", "cython", "scikit-learn", "numba", "inflect", "tqdm", 
        "anyascii", "pyyaml", "fsspec", "aiohttp", "packaging", "unidecode",
        "pysbd", "coqpit", "trainer", "soundfile", "librosa", "num2words",
        "tokenizers==0.15.2", "mecab-python3", "konlpy", "fugashi", "ipadic",
        "pandas<2.0", "umap-learn", "matplotlib"
    )
    .pip_install(
        # Ngôn ngữ bổ trợ (VI, CN, KR, etc.)
        "vinorm==2.0.7", "underthesea==6.8.4",
        "jieba", "pypinyin", "hangul_romanize", "jamo", "nltk", "g2pkk",
        "bangla", "bnnumerizer", "bnunicodenormalizer", "encodec",
        "spacy[ja]>=3", "sudachipy", "sudachidict_core"
    )
    .run_commands(f"rm -rf {CODE_DIR} && git clone {REPO_URL} {CODE_DIR}")
    .env({"PYTHONPATH": CODE_DIR})
)

# 5. Định nghĩa Class xử lý TTS
@app.cls(
    gpu="L4", 
    image=image,
    memory=16384,  # 16GB RAM cho CPU snapshot
    timeout=180,
    max_containers=1,
    volumes={"/cache": tts_cache},
    scaledown_window=180,
    enable_memory_snapshot=True,
    # Tắt GPU snapshot để dùng CPU snapshot ổn định hơn
)
class XTTSFTAPI:
    
    model_on_gpu = False

    @modal.enter(snap=True)
    def load_model_to_cpu(self):
        """Giai đoạn Snapshot: Nạp model vào RAM CPU"""
        import sys
        import gc
        
        os.environ['COQUI_TOS_AGREED'] = '1'
        if CODE_DIR not in sys.path:
            sys.path.append(CODE_DIR)
            
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        print("📥 Snapshot: Loading XTTS to CPU RAM...")
        
        # --- Monkey Patching để ép load vào CPU ---
        original_load = torch.load
        def cpu_load(*args, **kwargs):
            kwargs['map_location'] = torch.device('cpu')
            return original_load(*args, **kwargs)
        torch.load = cpu_load

        xtts_checkpoint = f"{MODEL_DIR}/checkpoint_686.pth"
        xtts_config = f"{MODEL_DIR}/config.json"
        xtts_vocab = f"{MODEL_DIR}/vocab.json"
        
        config = XttsConfig()
        config.load_json(xtts_config)
        
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, 
            checkpoint_path=xtts_checkpoint, 
            vocab_path=xtts_vocab, 
            use_deepspeed=False
        )
        self.model.eval()
        self.model.to("cpu")
        
        torch.load = original_load # Trả lại nguyên trạng
        
        self.speaker_wav_path = "/cache/refs/chunk_0004.wav"
        self.is_interrupted = False
        self.model_on_gpu = False
        
        gc.collect()
        print("✅ Model loaded in CPU RAM and snapshotted!")

    def move_to_gpu_if_needed(self):
        """Giai đoạn Runtime: Chuyển sang GPU khi cần thiết"""
        if self.model_on_gpu:
            return

        print("🚀 Runtime: Moving model to GPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            t0 = time.time()
            self.model.to(device)
            print(f"✅ Moved to GPU in {time.time() - t0:.2f}s")
            self.model_on_gpu = True
        else:
            print("⚠️ GPU not available, staying on CPU.")

    def _get_conditioning_latents(self, ref_path: str):
        """Helper để lấy đặc trưng giọng nói"""
        self.move_to_gpu_if_needed() # Đảm bảo model đã trên GPU
        
        if not ref_path or not os.path.exists(ref_path):
            print(f"⚠️ Warning: Reference audio {ref_path} not found.")
            return None, None
            
        return self.model.get_conditioning_latents(
            audio_path=[ref_path],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

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
    def tts_stream(self, body: Dict[str, Any]) -> StreamingResponse:
        # CHUYỂN QUA NGAY: Đảm bảo model lên GPU ngay khi nhận request đầu tiên
        self.move_to_gpu_if_needed()
        
        text = body.get("text", "")
        language = body.get("language", "en")
        
        if not text:
            return {"error": "Missing 'text' in request body"}
            
        def generate_audio_stream():
            ref_path = self.speaker_wav_path if os.path.exists(self.speaker_wav_path) else None
            gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(ref_path)
            
            # Tách chuỗi ngay tại Backend
            def split_text_smart(text_input, min_words=10, max_words=25):
                # 0. Loại bỏ thẻ HTML iframe, div và nội dung bên trong
                text_input = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text_input, flags=re.IGNORECASE | re.DOTALL)
                text_input = re.sub(r'<div[^>]*>.*?</div>', '', text_input, flags=re.IGNORECASE | re.DOTALL)
                
                # Loại bỏ các block code (nếu có để không đọc code)
                text_input = re.sub(r'```.*?```', '', text_input, flags=re.DOTALL)

                # 1. Loại bỏ hình ảnh markdown: ![alt](url)
                text_input = re.sub(r'!\[.*?\]\([^)]+\)', '', text_input)
                
                # 2. Thay thế link markdown: [text](url) -> giữ lại chữ "text", bỏ link
                text_input = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text_input)
                
                # 3. Loại bỏ các đoạn URL trần (http, https, www)
                text_input = re.sub(r'(https?://\S+|www\.\S+)', '', text_input)
                
                # 4. Loại bỏ dấu backtick của code block (``)
                text_input = re.sub(r'`+', '', text_input)

                # 5. Loại bỏ các ký tự Markdown không cần thiết (như **, #, _, ~)
                text_input = re.sub(r'[*#_~]', '', text_input)
                
                # 6. Thay thế các dấu gạch ngang (em dash —, en dash –, hyphen -) thành dấu phẩy để AI nghỉ nhịp
                text_input = re.sub(r'[—–-]', ',', text_input)
                
                # 7. Thay thế dấu xuống dòng (\n, \r) thành dấu chấm để ngắt câu (RẤT QUAN TRỌNG: để đọc thơ không bị dính chữ)
                text_input = text_input.replace('\r', '')
                text_input = re.sub(r'\n+', '. ', text_input)
                
                # 8. Dọn dẹp khoảng trắng thừa do việc xóa chữ để lại
                text_input = re.sub(r'\s+', ' ', text_input).strip()
                
                # Tách thô bằng dấu chấm, hỏi, than
                raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_input) if s.strip()]
                if not raw_sentences:
                    raw_sentences = [text_input]

                refined_chunks = []
                for sentence in raw_sentences:
                    if len(sentence.split()) > max_words:
                        # Tách phụ bằng dấu phẩy nếu quá dài
                        sub = [s.strip() for s in re.split(r'(?<=,)\s+', sentence) if s.strip()]
                        refined_chunks.extend(sub)
                    else:
                        refined_chunks.append(sentence)

                chunks = []
                buffer = ""
                for i, chunk in enumerate(refined_chunks):
                    candidate = (buffer + " " + chunk).strip()
                    
                    # Tính toán dựa trên số từ (words) chứ không phải char
                    words_count = len(candidate.split())
                    
                    # Mức độ ưu tiên min_words
                    if words_count < min_words and len(refined_chunks) > 1 and i == len(refined_chunks) - 1:
                        if chunks:
                            chunks[-1] = chunks[-1] + " " + candidate
                            buffer = ""
                            continue
                            
                    if words_count < min_words and i != len(refined_chunks) - 1:
                        buffer = candidate
                    else:
                        chunks.append(candidate)
                        buffer = ""
                        
                return chunks if chunks else [text_input.strip()]

            text_chunks = split_text_smart(text)
            self.is_interrupted = False
            
            print(f"📦 [Backend CHUNKS Tổng]: {text_chunks}")
            
            for chunk_text in text_chunks:
                if getattr(self, "is_interrupted", False):
                    print("🛑 Đã ngắt Inference (Đầu câu).")
                    break

                if not chunk_text.strip():
                    continue
                    
                # Vì XTTSv2 có tầng token nội bộ tự động cắt câu (in ra "> Text splitted to sentences.")
                # nên nó phớt lờ param tắt splitting, ta xử lý bằng cách thay các dấu ngắt câu thành dấu phẩy
                # Bắt luôn cả dấu xuống dòng \n nếu có lọt qua để XTTS không có lý do chia.
                chunk_for_xtts = re.sub(r'[.!?\n]', ',', chunk_text)
                
                print(f"🎤 [XTTS] Text đang truyền vào stream: '{chunk_for_xtts}'")
                
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
