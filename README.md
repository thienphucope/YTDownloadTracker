## speechGen — Dataset Pipeline

---

## Structure

```
src/getData/
├── downloadYouTube.py
├── importLocal.py
└── qualityPreset.py

downloads/{dataset_name}/
├── wavs/
├── videos/
├── subtitles/
└── getData.json
```

---

## downloadYouTube.py

```bash
# Audio
uv run python -m src.getData.downloadYouTube download <url> -d <dataset> -a <preset>

# Video
uv run python -m src.getData.downloadYouTube download <url> -d <dataset> -v <preset>

# Check files on disk
uv run python -m src.getData.downloadYouTube check -d <dataset>
```

Audio presets `-a`: `best` (44k) `high` (24k) `medium` (22.5k) `low` (16k)

Video presets `-v`: `max` `best` (1080p) `high` (720p) `medium` (480p) `low` (360p)

Optional: `--nodejs-path <path>`

---

## importLocal.py

```bash
uv run python -m src.getData.importLocal <wav_dir> -d <dataset>
uv run python -m src.getData.importLocal <wav_dir> -d <dataset> -o <output_dir>
```

Supported: `.wav` `.mp3` `.flac` `.ogg` `.m4a` `.opus`

---

See `libs/getData.md` for metadata structure and pipeline details.


