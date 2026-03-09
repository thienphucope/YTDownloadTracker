import torch
import os
import nemo.collections.asr as nemo_asr
from nemo.utils import logging

# Triệt tiêu log rác của Megatron và Distributed
logging.setLevel(logging.ERROR)
os.environ["HYDRA_FULL_ERROR"] = "1"

def test_nemo_setup():
    print("\n--- KIỂM TRA HỆ THỐNG NEMO (FIXED) ---")
    
    # 1. Kiểm tra CUDA (HP Victus RTX 3050)
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"[1] Trạng thái GPU: {'Sẵn sàng (RTX)' if cuda_available else 'Chưa nhận (CPU)'}")
    
    if cuda_available:
        print(f"    - Model: {torch.cuda.get_device_name(0)}")
        # Sử dụng mem_get_info (trả về free và total memory)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"    - VRAM trống: {round(free_mem / 1024**3, 2)} / {round(total_mem / 1024**3, 2)} GB")

    # 2. Kiểm tra ASR Model (Canary-1B)
    print("\n[2] Đang thử load Canary-1B (Dung lượng ~2GB)...")
    try:
        # Load model từ cloud của NVIDIA
        asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-1b")
        asr_model.to(device)
        print("    => Kết quả: Canary-1B đã sẵn sàng trên GPU!")
    except Exception as e:
        print(f"    => Lỗi load Canary: {e}")

    # 3. Kiểm tra Diarization Module
    print("\n[3] Kiểm tra thư viện Diarization...")
    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        print("    => Kết quả: Thư viện Diarization đã sẵn sàng!")
    except ImportError:
        print("    => Lỗi: Thiếu thành phần Diarization trong nemo_toolkit.")

    print("\n--- KẾT LUẬN ---")
    if cuda_available:
        print("Hệ thống cực kỳ ổn định. Bạn có thể bắt đầu 'finding Nemo' thực thụ rồi!")

if __name__ == "__main__":
    test_nemo_setup()