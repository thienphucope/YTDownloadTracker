import modal
import torch
from transformers import AutoTokenizer, pipeline
from typing import Dict, Any, List

# Tạo app Modal
app = modal.App("gemma-sealion-v4")

# Volume persistent cho HF cache (tạo bằng: modal volume create hf-cache)
hf_cache = modal.Volume.from_name("hf-cache")

# Build image với tokenizers pin (fix parsing error)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Để HF clone nếu cần
    .pip_install(
        "torch==2.4.1",  # Hỗ trợ NumPy 2.x & Python 3.11
        "transformers>=4.50.0",  # Cập nhật để hỗ trợ Gemma 3 architecture
        "accelerate==1.0.0",  # Stable tương thích
        "huggingface_hub>=0.24.6",  # Cập nhật để tương thích
        "tokenizers",  # Không pin nữa, dùng version mới
        "sentencepiece",  # Thêm SentencePiece để fix GemmaTokenizer error
        "fastapi",
    )
)

# Class để load và cache model
@app.cls(
    gpu="A100-80GB",  # Nâng cấp lên H200 cho tốc độ cao và VRAM lớn
    image=image,
    timeout=1800,  # 30 phút cho load/inference
    max_containers=1,  # Giới hạn 1 container cho GPU
    volumes={"/root/.cache/huggingface": hf_cache},  # Persistent cache
    container_idle_timeout=300,  # Giữ container idle 5 phút trước khi shutdown (tùy chọn để warm-up lâu hơn)
)
class GemmaSealion:
    @modal.enter()  # Setup: load model một lần khi container start
    def load_model(self):
        print("Loading model... (cached via volume)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "aisingapore/Gemma-SEA-LION-v4-27B-IT",
            cache_dir="/root/.cache/huggingface",
            trust_remote_code=True,
            use_fast=False,  # Fallback slow tokenizer để fix parsing error
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model="aisingapore/Gemma-SEA-LION-v4-27B-IT",
            tokenizer=self.tokenizer,  # Reuse tokenizer
            torch_dtype=torch.bfloat16,  # Theo model card: bfloat16 tốt hơn cho A100
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded and cached!")

    @modal.exit()  # Cleanup tùy chọn
    def cleanup(self):
        print("Cleaning up...")
        pass

    @modal.method()  # Method để generate với chat template chuẩn HF
    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 100) -> str:
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,  # Chỉ trả về phần generated mới
        )
        generated_text = result[0]["generated_text"].strip()
        return generated_text

    # Web endpoint trực tiếp trên class (POST / với JSON {"prompt": "...", "max_new_tokens": 100})
    # Chạy trên cùng container GPU, không cần .remote(), giảm latency đáng kể
    @modal.fastapi_endpoint(method="POST")
    def generate_text(self, body: Dict[str, Any]) -> Dict[str, str]:
        prompt = body.get("prompt", "")
        max_new_tokens = int(body.get("max_new_tokens", 100))
        if not prompt:
            return {"error": "Missing 'prompt' in request body"}

        messages = [
            {"role": "user", "content": prompt}
        ]
        result = self.generate.local(messages=messages, max_new_tokens=max_new_tokens)
        return {"result": result}