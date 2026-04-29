#!/usr/bin/env python3
"""
Local Audio Ingester
Scans a folder of existing audio files and creates dataset metadata.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

SUPPORTED_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}


def fmt_duration(secs) -> Optional[str]:
    if secs is None:
        return None
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _dumps_compact(obj) -> str:
    raw = json.dumps(obj, indent=2, ensure_ascii=False)
    compact = lambda s: ' '.join(s.split())
    for _ in range(4):
        raw = re.sub(r'\{([^{}\[\]]*)\}', lambda m: '{' + compact(m.group(1)) + '}', raw, flags=re.DOTALL)
    for _ in range(4):
        raw = re.sub(r'\[([^\[\]{}]*)\]', lambda m: '[' + compact(m.group(1)) + ']', raw, flags=re.DOTALL)
    return raw


class WavIngester:

    def __init__(self, dataset_name: str, wav_dir: str, output_dir: str):
        self.dataset_name = dataset_name
        self.wav_dir = Path(wav_dir).resolve()
        self.output_dir = Path(output_dir)
        self.metadata_path = self.output_dir / 'getData.json'

        self.data, self.meta = self._load()
        self._init_meta(dataset_name)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self):
        if self.metadata_path.exists():
            raw = json.loads(self.metadata_path.read_text(encoding='utf-8'))
            meta = raw.pop('_meta', {})
            return raw, meta
        return {}, {}

    def _save(self):
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        out = {'_meta': self.meta, **self.data}
        self.metadata_path.write_text(_dumps_compact(out), encoding='utf-8')

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def _init_meta(self, dataset_name: str):
        self.meta.setdefault('dataset_name', dataset_name)
        self.meta.setdefault('created_at', datetime.now().isoformat())
        self.meta.setdefault('last_updated_at', datetime.now().isoformat())
        sources = self.meta.setdefault('sources', [])
        if str(self.wav_dir) not in sources:
            sources.append(str(self.wav_dir))

    def refresh_stats(self):
        completed = [e for e in self.data.values() if e.get('status') == 'completed']

        durations = [e['duration'] for e in completed if e.get('duration')]
        avg_sec = sum(durations) / len(durations) if durations else None
        dur_stats = {
            'max_sec': max(durations) if durations else None,
            'min_sec': min(durations) if durations else None,
            'avg_sec': round(avg_sec, 1) if avg_sec else None,
            'max_hms': fmt_duration(max(durations)) if durations else None,
            'min_hms': fmt_duration(min(durations)) if durations else None,
            'avg_hms': fmt_duration(avg_sec) if avg_sec else None,
        }

        total_audio_sec  = round(sum(e.get('audio_properties', {}).get('audio_duration_sec', 0) for e in completed), 2)
        total_size_bytes = sum(e.get('file_size_bytes', 0) or 0 for e in completed)

        self.meta['last_updated_at'] = datetime.now().isoformat()
        self.meta.update({
            'total':                    len(self.data),
            'completed':                len(completed),
            'total_audio_duration_sec': total_audio_sec,
            'total_audio_duration_hms': fmt_duration(total_audio_sec),
            'total_file_size_bytes':    total_size_bytes,
            'total_file_size_mb':       round(total_size_bytes / (1024 * 1024), 2),
            'duration_stats':           dur_stats,
        })

    # ------------------------------------------------------------------
    # Ingest logic
    # ------------------------------------------------------------------

    def get_audio_properties(self, file_path: str) -> Dict:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', file_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            audio_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None
            )
            if not audio_stream:
                return {}
            fmt = data.get('format', {})
            duration = float(fmt.get('duration', 0))
            channels = int(audio_stream.get('channels', 0))
            bit_rate = int(audio_stream.get('bit_rate', 0) or fmt.get('bit_rate', 0) or 0)
            return {
                'sample_rate_hz':     int(audio_stream.get('sample_rate', 0)),
                'bit_depth':          int(audio_stream.get('bits_per_sample', 0)) or None,
                'channels':           channels,
                'is_mono':            channels == 1,
                'codec':              audio_stream.get('codec_name', ''),
                'bitrate_kbps':       bit_rate // 1000,
                'audio_duration_sec': round(duration, 2),
                'audio_duration_hms': fmt_duration(duration),
            }
        except Exception as e:
            print(f"⚠️  Could not extract audio properties: {e}")
            return {}

    def _build_entry(self, file_path: Path) -> Dict:
        now = datetime.now().isoformat()
        file_size = file_path.stat().st_size
        audio_props = self.get_audio_properties(str(file_path))
        duration = audio_props.get('audio_duration_sec')

        try:
            rel_path = str(file_path.relative_to(self.output_dir.parent))
        except ValueError:
            rel_path = str(file_path)

        return {
            'video_id':               None,
            'url':                    None,
            'source':                 'local',
            'title':                  file_path.stem,
            'channel':                None,
            'channel_id':             None,
            'upload_date':            None,
            'availability':           None,
            'live_status':            None,
            'age_limit':              None,
            'duration':               duration,
            'duration_hms':           fmt_duration(duration),
            'view_count':             None,
            'like_count':             None,
            'channel_follower_count': None,
            'description':            None,
            'audio_preset':           None,
            'video_preset':           None,
            'status':                 'completed',
            'count':                  len(self.data) + 1,
            'created_at':             now,
            'updated_at':             now,
            'retry_count':            0,
            'error':                  None,
            'file_path':              rel_path,
            'file_size_bytes':        file_size,
            'file_size_mb':           round(file_size / (1024 * 1024), 2),
            'audio_properties':       audio_props,
            'subtitle_files':         {},
        }

    def file_exists(self, file_id: str) -> bool:
        return self.data.get(file_id, {}).get('status') == 'completed'

    def ingest(self):
        files = sorted(
            f for f in self.wav_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )

        if not files:
            print(f"No audio files found in {self.wav_dir}")
            return

        print(f"Found {len(files)} audio file(s)")
        new_count = skipped = 0

        for f in files:
            file_id = f.stem
            if self.file_exists(file_id):
                print(f"⏭️  {file_id} — already ingested")
                skipped += 1
                continue

            print(f"  Processing {f.name}...")
            self.data[file_id] = self._build_entry(f)
            self._save()
            new_count += 1
            print(f"  ✅ {file_id}")

        self.refresh_stats()
        self._save()

        print(f"\nDone: {new_count} new, {skipped} skipped")
        print(f"Output: {self.metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Local Audio Ingester')
    parser.add_argument('wav_dir', help='Folder containing audio files')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset name')
    parser.add_argument('-o', '--output', help='Output directory (default: downloads/<dataset>)')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path('downloads') / args.dataset

    ingester = WavIngester(
        dataset_name=args.dataset,
        wav_dir=args.wav_dir,
        output_dir=str(output_dir),
    )

    try:
        ingester.ingest()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
