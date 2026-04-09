import modal
import torch
from typing import Dict, Any
from fastapi.responses import FileResponse, StreamingResponse
import io
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
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "transformers==4.37.2",
        "accelerate==0.27.2",
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
    timeout=180,
    max_containers=1,
    volumes={"/cache": tts_cache},
    scaledown_window=180,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class XTTSAPI:
    
    @modal.enter(snap=True)
    def load_model_to_gpu(self):
        """Chạy một lần khi tạo snapshot, load model thẳng vào VRAM"""
        os.environ['COQUI_TOS_AGREED'] = '1'
        self.is_interrupted = False
        
        from TTS.api import TTS
        print("🚀 Loading XTTS model to GPU...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model tts_v2
        self.tts = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(device)
        
        # Đường dẫn file mẫu trong Volume
        self.speaker_wav_path = "/cache/refs/schrolee.wav"
        
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
    def tts_interrupt(self) -> Dict[str, Any]:
        """Endpoint kích hoạt cờ báo ngắt Inference hiện tại."""
        self.is_interrupted = True
        print("🛑 Nhận được cờ Interrupt từ Frontend!")
        return {"status": "interrupted"}

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
        
    @modal.fastapi_endpoint(method="POST")
    def tts_stream(self, body: Dict[str, Any]) -> StreamingResponse:
        """Endpoint streaming trả về các chunk âm thanh liên tục."""
        text = body.get("text", "")
        language = body.get("language", "en")
        
        if not text:
            return {"error": "Missing 'text' in request body"}
            
        def generate_audio_stream():
            ref_path = self.speaker_wav_path if os.path.exists(self.speaker_wav_path) else None
            
            # Lấy conditioning latents từ mô hình XTTS gốc một lần duy nhất
            gpt_cond_latent, speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(audio_path=[ref_path])
            
            import numpy as np
            import re
            
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
                    
                # Gọi generator inference_stream
                audio_stream = self.tts.synthesizer.tts_model.inference_stream(
                    text=chunk_for_xtts,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding
                )
                
                accumulated_chunks = []
                for chunk in audio_stream:
                    if getattr(self, "is_interrupted", False):
                        print("🛑 Đã ngắt Inference (Giữa chừng).")
                        break

                    chunk_np = chunk.cpu().numpy() if torch.is_tensor(chunk) else np.array(chunk)
                    accumulated_chunks.append(chunk_np)
                    
                    # Gom 3 chunk nhỏ để cân bằng giữa giảm giật lag và độ trễ luồng
                    if len(accumulated_chunks) >= 3:
                        combined = np.concatenate(accumulated_chunks)
                        pcm_data = (combined * 32767).astype(np.int16).tobytes()
                        yield pcm_data
                        accumulated_chunks = []
                        
                # Trả về các chunk còn sót lại ở cuối mỗi câu
                if accumulated_chunks:
                    combined = np.concatenate(accumulated_chunks)
                    pcm_data = (combined * 32767).astype(np.int16).tobytes()
                    yield pcm_data
                
        return StreamingResponse(generate_audio_stream(), media_type="audio/pcm")

# Triển khai: modal deploy <tên_file>.py