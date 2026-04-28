## YouTube Dataset Downloader

A dataset-oriented YouTube downloader designed to:

- Ingest URLs — channel, playlist, or single video
- Extract unique `video_id` and avoid duplicate downloads
- Download audio (WAV) **or** video (MP4) per dataset
- Download subtitles with timestamps (VTT)
- Track all download state and metadata in `metadata.json`

---

## Folder Structure

```
downloads/{dataset_name}/
├── wavs/                        # audio mode only
│   └── {video_id}.wav
├── videos/                      # video mode only
│   └── {video_id}.mp4
├── subtitles/                   # always present
│   ├── {video_id}.{lang}.vtt
│   └── ...
└── metadata.json
```

---

## Usage

```bash
# Download audio
uv run python downloadYT.py https://www.youtube.com/@Channel -d my_dataset -a best

# Download video
uv run python downloadYT.py https://www.youtube.com/@Channel -d my_dataset -v eco

# Single video
uv run python downloadYT.py https://www.youtube.com/watch?v=VIDEO_ID -d single -a eco
```

`-a` and `-v` are mutually exclusive — one must be specified.

---

## Modes & Presets

### Audio (`-a`)

| Preset | Sample Rate | Bit Depth | Channels |
|--------|------------|-----------|----------|
| `best` | 24,000 Hz  | 16-bit    | Mono     |
| `eco`  | 16,000 Hz  | 16-bit    | Mono     |

Source audio is downloaded via yt_dlp then converted with ffmpeg to WAV at the target spec.

### Video (`-v`)

| Preset | Quality |
|--------|---------|
| `best` | Best available (yt_dlp selects highest quality MP4) |
| `eco`  | Capped at 720p |

Video includes audio natively (merged by yt_dlp). No post-processing step.

### Subtitles

Downloaded automatically alongside audio/video:
- **Manual subs** — video's primary language only
- **Auto captions** — `en` and `vi` only
- Format: VTT (with timestamps)

---

## Metadata Structure

`metadata.json` has two sections: a global `_meta` block and per-video entries keyed by `video_id`.

### Global `_meta`

```json
"_meta": {
  "dataset_name": "solus",
  "audio_preset": "best",
  "video_preset": null,
  "created_at": "...",
  "last_updated_at": "...",
  "sources": ["https://www.youtube.com/@Channel"],
  "stats": {
    "total": 50,
    "completed": 48,
    "pending": 0,
    "pending_ids": [],
    "failed": 2,
    "failed_ids": ["abc12345678", "xyz98765432"],
    "total_audio_duration_sec": 18450.5,
    "total_file_size_bytes": 524288000
  }
}
```

`sources` records every URL fed into the dataset. `failed_ids` and `pending_ids` let you find problem entries without iterating the full file.

### Per-video Entry

```json
"VIDEO_ID": {
  "video_id": "...",
  "url": "...",
  "title": "...",
  "channel": "...",
  "channel_id": "UC...",
  "upload_date": "2023-07-15",
  "availability": "public",
  "live_status": "not_live",
  "age_limit": 0,
  "duration": 847.0,
  "view_count": 125000,
  "like_count": 4200,
  "channel_follower_count": 80000,
  "audio_preset": "best",
  "video_preset": null,
  "description": "...",
  "status": "completed",
  "created_at": "...",
  "updated_at": "...",
  "last_checked_at": "...",
  "file_path": "solus/wavs/VIDEO_ID.wav",
  "file_size_bytes": 10485760,
  "audio_properties": {
    "sample_rate_hz": 24000,
    "bit_depth": 16,
    "channels": 1,
    "is_mono": true,
    "codec": "pcm_s16le",
    "bitrate_kbps": 384,
    "audio_duration_sec": 847.02
  },
  "subtitle_files": {
    "vi": "solus/subtitles/VIDEO_ID.vi.vtt",
    "en": "solus/subtitles/VIDEO_ID.en.vtt"
  },
  "retry_count": 0,
  "error": null,
  "count": 1
}
```

`file_path` and `subtitle_files` paths are relative to the `downloads/` parent — portable across machines.


