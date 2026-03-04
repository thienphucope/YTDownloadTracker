Dưới đây là bản README tóm tắt đúng scope **Downloader + Tracking Only** cho dataset audio:

---

## 📦 YouTube Audio Dataset Downloader

This project is a dataset-oriented YouTube audio downloader designed to:

* Ingest video URLs such as channel, playlist or single video link
* Extract unique `video_id`
* Avoid duplicate downloads
* Store extracted audio into dataset-specific folders
* Track download state and audio properties via metadata

Each dataset is treated as a **collection**, where downloaded audio and metadata are grouped under:

```
downloads/{dataset_name}/
├── wavs/
│   ├── {video_id}.wav
│   └── ...
└── metadata.json
```

---

## 🔁 Workflow

1. Input a YouTube video URL and choose or create a collection aka dataset name to add
2. Extract `video_id` from URL
3. Check if `video_id` exists in `metadata.json`

   * If exists and already downloaded → skip
   * Else → create the entry in the metadata.json and proceed to download 
4. Extract audio and store at:

   ```
   downloads/{dataset}/wavs/{video_id}.wav
   ```
5. Update `metadata.json` entry with:

   * Download status
   * File path
   * Audio specifications (sample rate, bit depth, channels, etc.)
   * Retry count and error (if failed)

---

## 🎧 Audio Quality Presets

You can choose from two audio quality presets when downloading:

*   **`best` (Default):** Provides high-quality audio for training purposes.
    *   **Format:** WAV (PCM)
    *   **Sample Rate:** 24,000 Hz
    *   **Bit Depth:** 16-bit
    *   **Channels:** Mono

*   **`eco`:** Reduces file size with minimal impact on perceptual quality, suitable for quick experiments or when storage is a concern.
    *   **Format:** WAV (PCM)
    *   **Sample Rate:** 16,000 Hz
    *   **Bit Depth:** 16-bit
    *   **Channels:** Mono

---

## 🧾 Metadata Entry Includes

Each entry in `metadata.json` tracks:

* Video identity (`video_id`, `url`, `title`, `channel`, `description`)
* Download status and timestamps
* File path and size
* Audio properties:

  * `sample_rate_hz`
  * `bit_depth`
  * `channels`
  * `is_mono`
  * `codec`
  * `bitrate_kbps`
  * `audio_duration_sec`
* Error tracking (`status`, `retry_count`, `error`)
* Sequential `count` value (added on entry creation) to show how many
  items have been recorded without having to iterate the file

---

This metadata serves as the single source of truth for dataset tracking and prevents redundant downloads.

---

## 💡 Usage

Expected command:

```bash
uv run python .\main.py https://www.youtube.com/@SolusAstorias -d solus -q best
```

**Note:** Using `-d solus` will automatically create the dataset folder at `downloads/solus/` with the default structure.