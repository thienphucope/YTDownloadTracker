import os
import torch
import warnings
import soundfile as sf
import tempfile
from flask import Flask, render_template_string, request, jsonify

# Triệt tiêu log hệ thống
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

from pyannote.audio import Pipeline

app = Flask(__name__)

# --- CẤU HÌNH ---
HF_TOKEN = ""

# Tối ưu cho GPU RTX
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1. Load pipeline 3.1
print("--- Đang tải model Pyannote 3.1... ---")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
pipeline.to(torch.device("cuda"))

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speaker Diarization Visualizer</title>
        <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
        <script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 30px; }
            #waveform { background: #1e1e1e; border: 1px solid #333; border-radius: 8px; margin: 20px 0; }
            .controls { margin-bottom: 20px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
            .file-input-wrapper { background: #333; padding: 8px; border-radius: 5px; border: 1px dashed #555; }
            button { padding: 10px 25px; cursor: pointer; background: #007bff; border: none; color: white; border-radius: 5px; font-weight: bold; transition: 0.2s; }
            button:disabled { background: #444; cursor: not-allowed; }
            button:hover:not(:disabled) { background: #0056b3; }
            #status { color: #00ffcc; font-weight: bold; }
            
            .legend { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px; }
            .legend-item { 
                padding: 6px 15px; border-radius: 20px; font-size: 13px; color: #fff; 
                font-weight: bold; cursor: pointer; user-select: none; transition: 0.3s;
                border: 2px solid transparent; opacity: 1;
            }
            .legend-item.hidden {
                opacity: 0.3;
                background-color: #444 !important;
                text-decoration: line-through;
            }
        </style>
    </head>
    <body>
        <h1>AI Speaker Diarization WebUI (3.1)</h1>
        
        <div class="controls">
            <div class="file-input-wrapper">
                <input type="file" id="audioFile" accept="audio/*">
            </div>
            <button id="runBtn" onclick="processAudio()">RUN DIARIZATION</button>
            <button onclick="wavesurfer.playPause()">PLAY / PAUSE</button>
            <span id="status">Sẵn sàng.</span>
        </div>

        <div id="waveform"></div>
        <p style="font-size: 12px; color: #888;">* Click vào tên Speaker bên dưới để ẩn/hiện vùng màu tương ứng trên track.</p>
        <div class="legend" id="legend"></div>

        <script>
            let currentBlobUrl = null;
            let allSegments = []; // Lưu trữ kết quả gốc
            const wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: '#555',
                progressColor: '#007bff',
                height: 150,
            });

            const wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());
            const colors = [
                'rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 
                'rgba(255, 206, 86, 0.5)', 'rgba(75, 192, 192, 0.5)',
                'rgba(153, 102, 255, 0.5)', 'rgba(255, 159, 64, 0.5)'
            ];
            
            const speakerVisibility = {}; // Lưu trạng thái ẩn/hiện của từng speaker
            const speakerColorMap = {};

            document.getElementById('audioFile').onchange = function(e) {
                const file = e.target.files[0];
                if (file) {
                    if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
                    currentBlobUrl = URL.createObjectURL(file);
                    wavesurfer.load(currentBlobUrl);
                    document.getElementById('status').innerText = "Đã tải: " + file.name;
                    wsRegions.clearRegions();
                    document.getElementById('legend').innerHTML = '';
                    allSegments = [];
                }
            };

            // Hàm vẽ lại các vùng dựa trên trạng thái ẩn hiện
            function redrawRegions() {
                wsRegions.clearRegions();
                allSegments.forEach(segment => {
                    if (speakerVisibility[segment.speaker]) {
                        wsRegions.addRegion({
                            start: segment.start,
                            end: segment.end,
                            content: segment.speaker,
                            color: speakerColorMap[segment.speaker],
                            drag: false, resize: false
                        });
                    }
                });
            }

            function toggleSpeaker(speaker, element) {
                speakerVisibility[speaker] = !speakerVisibility[speaker];
                element.classList.toggle('hidden', !speakerVisibility[speaker]);
                redrawRegions();
            }

            async function processAudio() {
                const fileInput = document.getElementById('audioFile');
                const status = document.getElementById('status');
                const runBtn = document.getElementById('runBtn');

                if (!fileInput.files[0]) {
                    alert("Vui lòng chọn file!"); return;
                }

                const formData = new FormData();
                formData.append('audio', fileInput.files[0]);
                status.innerText = "Processing on GPU...";
                runBtn.disabled = true;
                
                try {
                    const response = await fetch('/diarize', { method: 'POST', body: formData });
                    allSegments = await response.json();
                    
                    const legend = document.getElementById('legend');
                    legend.innerHTML = '';
                    
                    let colorIdx = 0;
                    allSegments.forEach(s => {
                        if (!speakerColorMap[s.speaker]) {
                            speakerColorMap[s.speaker] = colors[colorIdx % colors.length];
                            speakerVisibility[s.speaker] = true; // Mặc định là hiện
                            colorIdx++;

                            const div = document.createElement('div');
                            div.className = 'legend-item';
                            div.style.backgroundColor = speakerColorMap[s.speaker];
                            div.innerText = s.speaker;
                            div.onclick = () => toggleSpeaker(s.speaker, div);
                            legend.appendChild(div);
                        }
                    });

                    redrawRegions();
                    status.innerText = "Done!";
                } catch (e) {
                    status.innerText = "Error!";
                    console.error(e);
                } finally {
                    runBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """)

@app.route('/diarize', methods=['POST'])
def do_diarization():
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        data, samplerate = sf.read(tmp_path)
        waveform = torch.tensor(data).float().t()
        if len(waveform.shape) == 1: waveform = waveform.unsqueeze(0)
        
        audio_data = {"waveform": waveform, "sample_rate": samplerate}
        print(f"--- Đang xử lý Diarization trên GPU ---")
        output = pipeline(audio_data)
        
        results = [{"start": round(t.start, 2), "end": round(t.end, 2), "speaker": s} 
                   for t, s in output.speaker_diarization]
        return jsonify(results)
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)