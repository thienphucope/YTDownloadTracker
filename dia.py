import os
import json
import shutil
import torch
import gc
from flask import Flask, render_template_string, request, jsonify
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecMultiTaskModel, ClusteringDiarizer

app = Flask(__name__)

# --- KHỞI TẠO THƯ MỤC ---
BASE_OUT_DIR = os.path.join(os.getcwd(), "nemo_outputs")
if not os.path.exists(BASE_OUT_DIR):
    os.makedirs(BASE_OUT_DIR)

def get_oracle_diar_config(manifest_path, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cấu hình "Full Giáp" cho Oracle Diarization
    return OmegaConf.create({
        'name': "ClusterDiarizer",
        'device': device,
        'batch_size': 16,
        'num_workers': 0,
        'diarizer': {
            'manifest_filepath': manifest_path,
            'out_dir': out_dir,
            'oracle_vad': True, # Dùng timestamp dẫn đường từ ASR
            'collar': 0.25,
            'ignore_overlap': True,
            
            # ĐÃ FIX: Key 'vad' rỗng bắt buộc cho Oracle Mode
            'vad': {
                'model_path': None,
                'external_vad_manifest': None,
                'parameters': {}
            },
            
            'speaker_embeddings': {
                'model_path': 'titanet_large',
                'parameters': {
                    'window_length_in_sec': [1.5, 1.0, 0.5],
                    'shift_length_in_sec': [0.75, 0.5, 0.25],
                    'multiscale_weights': [1, 1, 1],
                    'save_embeddings': False,
                    'chunk_batch_size': 64
                }
            },
            
            'clustering': {
                'parameters': {
                    'oracle_num_speakers': False,
                    'max_num_speakers': 20,
                    'enhanced_mag_threshold': 1,
                    'sparse_search_volume': 30,
                    'max_rp_threshold': 0.25,
                    'sparse_search': True,
                    'cholesky_threshold': 1e-3,
                    'min_num_speakers': 1
                }
            }
        }
    })

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NeMo SOTA: Canary + Oracle Diarization</title>
        <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
        <script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
        <style>
            body { font-family: 'Inter', sans-serif; background: #0b0b0b; color: #fff; padding: 40px; }
            .container { max-width: 1100px; margin: 0 auto; background: #161616; padding: 30px; border-radius: 15px; border: 1px solid #333; }
            #waveform { background: #000; border-radius: 10px; margin: 25px 0; border: 1px solid #222; }
            .controls { display: flex; align-items: center; gap: 15px; background: #222; padding: 20px; border-radius: 10px; }
            .btn { padding: 12px 25px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: 0.2s; }
            .btn-run { background: #ef4444; color: white; }
            .btn:disabled { background: #444; opacity: 0.6; }
            #status { color: #facc15; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
            .transcript-area { background: #111; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 25px; max-height: 400px; overflow-y: auto; }
            .segment { border-bottom: 1px solid #262626; padding: 15px 0; }
            .spk-id { font-weight: bold; padding: 4px 12px; border-radius: 6px; margin-right: 12px; font-size: 12px; color: #000; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎙️ SOTA: Canary-1B + Oracle Diarization</h2>
            <div class="controls">
                <input type="file" id="audioFile" accept="audio/*">
                <button class="btn btn-run" id="runBtn" onclick="runPipeline()">START PROCESSING</button>
                <span id="status">Ready to go.</span>
            </div>
            <div id="waveform"></div>
            <div class="transcript-area" id="transcript">Transcript results will appear here...</div>
        </div>

        <script>
            const wavesurfer = WaveSurfer.create({ container: '#waveform', waveColor: '#333', progressColor: '#ef4444', height: 160 });
            const wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());
            const spkColors = ['#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899'];
            let colorCache = {};

            document.getElementById('audioFile').onchange = (e) => wavesurfer.load(URL.createObjectURL(e.target.files[0]));

            async function runPipeline() {
                const file = document.getElementById('audioFile').files[0];
                if(!file) return;
                const fd = new FormData(); fd.append('audio', file);
                const btn = document.getElementById('runBtn');
                const status = document.getElementById('status');
                
                btn.disabled = true;
                status.innerText = "Phase 1: Canary-1B Transcribing (VRAM check)...";
                wsRegions.clearRegions();

                try {
                    const res = await fetch('/process', { method: 'POST', body: fd });
                    const data = await res.json();
                    if(data.error) throw new Error(data.error);

                    status.innerText = "Success! Displaying Timeline.";
                    let html = ""; let cIdx = 0;

                    data.results.forEach(seg => {
                        if(!colorCache[seg.speaker]) {
                            colorCache[seg.speaker] = spkColors[cIdx % spkColors.length];
                            cIdx++;
                        }
                        const color = colorCache[seg.speaker];
                        wsRegions.addRegion({ start: seg.start, end: seg.end, color: color + '55', content: seg.speaker, drag: false, resize: false });
                        
                        html += `<div class="segment">
                            <span class="spk-id" style="background:${color}">${seg.speaker}</span>
                            <small style="color:#666">[${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s]</small>
                            <p style="margin-top:8px; color:#ddd;">${data.full_text}</p>
                        </div>`;
                    });
                    document.getElementById('transcript').innerHTML = html;
                } catch (e) {
                    status.innerText = "Error: " + e.message;
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """)

@app.route('/process', methods=['POST'])
def process_full_pipeline():
    audio_file = request.files['audio']
    if os.path.exists(BASE_OUT_DIR):
        try: shutil.rmtree(BASE_OUT_DIR)
        except: pass
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    
    audio_path = os.path.join(BASE_OUT_DIR, "input.wav")
    audio_file.save(audio_path)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- PHASE 1: CANARY-1B STT ---
        print("--- Loading Canary-1B STT ---")
        asr_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').to(device)
        
        print("--- Transcribing Audio ---")
        transcripts = asr_model.transcribe([audio_path])
        # ĐÃ FIX: Ép kiểu string cho Hypothesis object
        final_text = str(transcripts[0]) if isinstance(transcripts, list) else str(transcripts)

        # Giải phóng VRAM RTX 3050 ngay lập tức
        del asr_model
        gc.collect()
        torch.cuda.empty_cache()

        # Tạo manifest chuẩn Oracle (Bổ sung rttm/uem)
        manifest_path = os.path.join(BASE_OUT_DIR, "oracle_manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "audio_filepath": audio_path, "offset": 0, "duration": None, 
                "label": "speech", "text": final_text, "num_speakers": None,
                "rttm_filepath": None, "uem_filepath": None
            }) + "\n")

        # --- PHASE 2: ORACLE DIARIZATION ---
        print("--- Loading TitaNet Diarizer ---")
        cfg = get_oracle_diar_config(manifest_path, BASE_OUT_DIR)
        sd_model = ClusteringDiarizer(cfg=cfg).to(device)
        
        print("--- Running Diarization ---")
        sd_model.diarize()

        # Dọn dẹp VRAM
        del sd_model
        gc.collect()
        torch.cuda.empty_cache()

        # --- PHASE 3: PARSE RESULTS ---
        rttm_path = os.path.join(BASE_OUT_DIR, "pred_rttms", "input.rttm")
        results = []
        if os.path.exists(rttm_path):
            with open(rttm_path, 'r') as f:
                for line in f:
                    p = line.split()
                    results.append({
                        "start": float(p[3]), "end": float(p[3]) + float(p[4]), "speaker": p[7]
                    })
        
        return jsonify({"full_text": final_text, "results": results})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Tắt reloader để tránh Flask restart khi NeMo tải file tạm
    app.run(debug=True, use_reloader=False, port=5000)