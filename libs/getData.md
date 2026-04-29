# getData.json — Cách hoạt động

`getData.json` là file metadata của bước đầu pipeline (download hoặc import local). Mỗi dataset có một file riêng tại `downloads/{dataset_name}/getData.json`.

---

## Cấu trúc JSON

```json
{
  "_meta": { ... },
  "<id>": { ... },
  "<id>": { ... }
}
```

---

## `_meta` — Dataset level

Flat, không wrapper:

```json
"_meta": {
  "dataset_name": "my_dataset",
  "created_at": "2026-04-29T00:00:00",
  "last_updated_at": "2026-04-29T01:00:00",
  "sources": ["https://www.youtube.com/@Channel"],

  "total": 50,
  "completed": 48,
  "pending": 0,  "pending_ids": [],
  "failed": 2,   "failed_ids": ["abc123"],

  "total_audio_duration_sec": 18450.5,
  "total_audio_duration_hms": "05:07:30",
  "total_file_size_bytes": 524288000,
  "total_file_size_mb": 500.0,

  "duration_stats": {"max_sec": 3612, "min_sec": 124, "avg_sec": 843.5, "max_hms": "01:00:12", "min_hms": "00:02:04", "avg_hms": "00:14:03"},
  "has_subtitles": 45,
  "no_subtitles": 3,
  "subtitle_languages": {"vi": 45, "en": 30},
  "date_range": {"oldest": "2021-01-10", "newest": "2024-12-01"}
}
```

---

## Entry — Per item

Flat hoàn toàn, không group:

```json
"VIDEO_ID": {
  "video_id": "VIDEO_ID",
  "url": "https://...",
  "source": "https://www.youtube.com/@Channel",
  "title": "...",
  "channel": "...",
  "channel_id": "UC...",
  "upload_date": "2023-07-15",
  "availability": "public",
  "live_status": "not_live",
  "age_limit": 0,
  "duration": 847,
  "duration_hms": "00:14:07",
  "view_count": 125000,
  "like_count": 4200,
  "channel_follower_count": 80000,
  "description": "...",

  "audio_preset": "high",
  "video_preset": null,
  "status": "completed",
  "count": 1,
  "created_at": "...",
  "updated_at": "...",
  "retry_count": 0,
  "error": null,

  "file_path": "my_dataset/wavs/VIDEO_ID.wav",
  "file_size_bytes": 10485760,
  "file_size_mb": 10.0,
  "audio_properties": {"sample_rate_hz": 24000, "bit_depth": 16, "channels": 1, "is_mono": true, "codec": "pcm_s16le", "bitrate_kbps": 384, "audio_duration_sec": 847.02, "audio_duration_hms": "00:14:07"},
  "video_properties": {},
  "subtitle_files": {"vi": "my_dataset/subtitles/VIDEO_ID.vi.vtt"}
}
```

`importLocal.py` dùng cùng schema — các field YouTube-specific là `null`, `source = "local"`.

---

## Pipeline

```
downloadYouTube.py / importLocal.py  →  getData.json  (owns hoàn toàn)
step2                                →  đọc getData.json, ghi step2.json
step3                                →  đọc getData.json + step2.json, ghi step3.json
```

Mỗi step chỉ đọc file của step trước, không bao giờ ghi đè.
