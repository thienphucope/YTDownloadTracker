#!/usr/bin/env python3
"""
YouTube Audio/Video Dataset Downloader
Downloads audio or video from YouTube URLs and tracks metadata for dataset management.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

from src.getData.qualityPreset import QualityPreset


def fmt_duration(secs) -> Optional[str]:
    if secs is None:
        return None
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _dumps_compact(obj) -> str:
    """JSON indent=2 but leaf objects and primitive arrays stay on one line."""
    raw = json.dumps(obj, indent=2, ensure_ascii=False)
    compact = lambda s: ' '.join(s.split())
    for _ in range(4):
        raw = re.sub(r'\{([^{}\[\]]*)\}', lambda m: '{' + compact(m.group(1)) + '}', raw, flags=re.DOTALL)
    for _ in range(4):
        raw = re.sub(r'\[([^\[\]{}]*)\]', lambda m: '[' + compact(m.group(1)) + ']', raw, flags=re.DOTALL)
    return raw


class YouTubeDownloader:
    """Downloads audio or video from YouTube and manages dataset."""

    def __init__(self, dataset_name: str, output_dir: str,
                 audio_quality: Optional[str] = None, video_quality: Optional[str] = None):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.audio_quality = audio_quality
        self.video_quality = video_quality
        self.metadata_path = self.output_dir / 'getData.json'

        self.subtitles_dir = self.output_dir / 'subtitles'
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)

        if audio_quality:
            self.wavs_dir = self.output_dir / 'wavs'
            self.wavs_dir.mkdir(parents=True, exist_ok=True)
        if video_quality:
            self.videos_dir = self.output_dir / 'videos'
            self.videos_dir.mkdir(parents=True, exist_ok=True)

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
        self.meta.setdefault('sources', [])

    def refresh_stats(self):
        completed   = [e for e in self.data.values() if e.get('status') == 'completed']
        pending_ids = [vid for vid, e in self.data.items() if e.get('status') == 'pending']
        failed_ids  = [vid for vid, e in self.data.items() if e.get('status') == 'failed']

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

        lang_count: Dict[str, int] = {}
        has_subs = 0
        for e in completed:
            subs = e.get('subtitle_files') or {}
            if subs:
                has_subs += 1
                for lang in subs:
                    lang_count[lang] = lang_count.get(lang, 0) + 1

        dates = sorted(e['upload_date'] for e in completed if e.get('upload_date'))

        self.meta['last_updated_at'] = datetime.now().isoformat()
        self.meta.update({
            'total':       len(self.data),
            'completed':   len(completed),
            'pending':     len(pending_ids),
            'pending_ids': pending_ids,
            'failed':      len(failed_ids),
            'failed_ids':  failed_ids,
            'total_audio_duration_sec': total_audio_sec,
            'total_audio_duration_hms': fmt_duration(total_audio_sec),
            'total_file_size_bytes': total_size_bytes,
            'total_file_size_mb':    round(total_size_bytes / (1024 * 1024), 2),
            'duration_stats':   dur_stats,
            'has_subtitles':    has_subs,
            'no_subtitles':     len(completed) - has_subs,
            'subtitle_languages': lang_count,
            'date_range':       {'oldest': dates[0], 'newest': dates[-1]} if dates else {},
        })

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    def video_exists(self, video_id: str) -> bool:
        return self.data.get(video_id, {}).get('status') == 'completed'

    def add_source(self, url: str):
        sources = self.meta.setdefault('sources', [])
        if url not in sources:
            sources.append(url)
            self._save()

    def mark_completed(self, video_id: str, file_path: str,
                       file_size: Optional[int], audio_props: Dict,
                       video_props: Optional[Dict] = None):
        e = self.data.get(video_id)
        if not e:
            return
        e.update(
            status='completed',
            updated_at=datetime.now().isoformat(),
            file_path=file_path,
            file_size_bytes=file_size,
            file_size_mb=round(file_size / (1024 * 1024), 2) if file_size else None,
            audio_properties=audio_props,
            video_properties=video_props or {},
        )
        self._save()

    def mark_failed(self, video_id: str, error: str):
        e = self.data.get(video_id)
        if not e:
            return
        e.update(
            status='failed',
            updated_at=datetime.now().isoformat(),
            error=str(error),
            retry_count=e.get('retry_count', 0) + 1,
        )
        self._save()

    def _build_entry(self, url: str, info: Dict, source: str = '') -> Dict:
        upload_date = info.get('upload_date', '')
        if upload_date and len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
        now = datetime.now().isoformat()
        duration = info.get('duration')
        return {
            'video_id':               info.get('id', ''),
            'url':                    url,
            'source':                 source,
            'title':                  info.get('title', ''),
            'channel':                info.get('uploader', ''),
            'channel_id':             info.get('channel_id', ''),
            'upload_date':            upload_date,
            'availability':           info.get('availability', 'public'),
            'live_status':            info.get('live_status', 'not_live'),
            'age_limit':              info.get('age_limit', 0),
            'duration':               duration,
            'duration_hms':           fmt_duration(duration),
            'view_count':             info.get('view_count'),
            'like_count':             info.get('like_count'),
            'channel_follower_count': info.get('channel_follower_count'),
            'description':            (info.get('description') or '')[:500],
            'audio_preset':           self.audio_quality,
            'video_preset':           self.video_quality,
            'status':                 'pending',
            'count':                  len(self.data) + 1,
            'created_at':             now,
            'updated_at':             now,
            'retry_count':            0,
            'error':                  None,
            'file_path':              None,
            'file_size_bytes':        None,
            'file_size_mb':           None,
            'audio_properties':       {},
            'video_properties':       {},
            'subtitle_files':         {},
        }

    # ------------------------------------------------------------------
    # Download logic
    # ------------------------------------------------------------------

    def extract_video_ids(self, url: str) -> List[str]:
        is_channel = '@' in url or '/c/' in url or '/channel/' in url or '/user/' in url

        urls_to_process = []
        if is_channel:
            base_url = url.rstrip('/')
            for tab in ['/videos', '/shorts', '/streams']:
                base_url = base_url.replace(tab, '')
            urls_to_process = [base_url + '/videos', base_url + '/shorts', base_url + '/streams']
        else:
            urls_to_process = [url]

        ydl_opts = {
            'quiet': False,
            'extract_flat': 'in_playlist',
            'skip_download': True,
            'ignoreerrors': True,
        }

        video_ids = []
        for current_url in urls_to_process:
            try:
                print(f"🔄 Processing: {current_url}")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(current_url, download=False)
                if not info:
                    continue
                if 'entries' in info:
                    for entry in info['entries']:
                        if entry and 'id' in entry:
                            if not entry['id'].startswith('UC') or len(entry['id']) != 24:
                                video_ids.append(entry['id'])
                            elif 'url' in entry:
                                match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&/]|$)', entry['url'])
                                if match:
                                    video_ids.append(match.group(1))
                elif 'id' in info:
                    video_ids.append(info['id'])
            except Exception as e:
                print(f"⚠️  Error extracting from {current_url}: {e}")

        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen and len(vid) == 11:
                seen.add(vid)
                unique_ids.append(vid)
        return unique_ids

    def get_video_info(self, video_id: str) -> Optional[Dict]:
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
                return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        except Exception as e:
            print(f"❌ Error getting video info for {video_id}: {e}")
            return None

    def download_audio(self, video_id: str) -> Optional[str]:
        output_path = self.wavs_dir / f"{video_id}.wav"
        temp_path = self.wavs_dir / f"{video_id}_temp.wav"
        converted_path = self.wavs_dir / f"{video_id}_converted.wav"

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(temp_path.with_suffix('.%(ext)s')),
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'quiet': False,
            'no_warnings': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            if not temp_path.exists():
                raise FileNotFoundError(f"Downloaded file not found: {temp_path}")
            print(f"📁 Downloaded: {temp_path.name} ({temp_path.stat().st_size / 1024:.1f} KB)")

            preset = QualityPreset.audio(self.audio_quality)
            print(f"🔄 Converting to {preset['sample_rate']}Hz mono...")
            self._convert_audio(temp_path, converted_path, preset)

            if not converted_path.exists():
                raise FileNotFoundError(f"Converted file not found: {converted_path}")

            if output_path.exists():
                output_path.unlink()
            converted_path.rename(output_path)
            print(f"📁 Final: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")

            if temp_path.exists():
                temp_path.unlink()

            if not output_path.exists():
                raise FileNotFoundError(f"Final file missing: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"❌ Audio download failed for {video_id}: {e}")
            import traceback
            traceback.print_exc()
            for pattern in [f"{video_id}_temp.*", f"{video_id}_converted.*"]:
                for f in self.wavs_dir.glob(pattern):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            return None

    def _convert_audio(self, input_path: Path, output_path: Path, preset: Dict):
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must be different")
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-ar', str(preset['sample_rate']),
            '-ac', str(preset['channels']),
            '-sample_fmt', f's{preset["bit_depth"]}',
            '-n', str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")

    def download_video(self, video_id: str) -> Optional[str]:
        output_path = self.videos_dir / f"{video_id}.mp4"
        preset = QualityPreset.video(self.video_quality)

        ydl_opts = {
            'format': preset['format'],
            'outtmpl': str(self.videos_dir / f'{video_id}.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            if not output_path.exists():
                raise FileNotFoundError(f"Video file not found: {output_path}")
            print(f"🎬 Video: {output_path.name} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
            return str(output_path)

        except Exception as e:
            print(f"❌ Video download failed for {video_id}: {e}")
            import traceback
            traceback.print_exc()
            for f in self.videos_dir.glob(f"{video_id}.*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            return None

    def download_subtitles(self, video_id: str, info: Dict) -> Dict[str, str]:
        url = f"https://www.youtube.com/watch?v={video_id}"
        base_opts = {
            'skip_download': True,
            'subtitlesformat': 'vtt',
            'outtmpl': str(self.subtitles_dir / f'{video_id}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        video_lang = info.get('language')
        if info.get('subtitles') and video_lang:
            try:
                with yt_dlp.YoutubeDL({
                    **base_opts,
                    'writesubtitles': True,
                    'writeautomaticsub': False,
                    'subtitleslangs': [video_lang],
                }) as ydl:
                    ydl.download([url])
            except Exception as e:
                print(f"⚠️  Manual subtitle download failed: {e}")

        if info.get('automatic_captions'):
            try:
                with yt_dlp.YoutubeDL({
                    **base_opts,
                    'writesubtitles': False,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', 'vi'],
                }) as ydl:
                    ydl.download([url])
            except Exception as e:
                print(f"⚠️  Auto caption download failed: {e}")

        subtitle_files = {}
        for f in self.subtitles_dir.glob(f"{video_id}.*.vtt"):
            lang = f.stem.split('.', 1)[1]
            subtitle_files[lang] = str(f.relative_to(self.output_dir.parent))

        if subtitle_files:
            print(f"📝 Subtitles downloaded: {', '.join(subtitle_files.keys())}")
        return subtitle_files

    def _ffprobe(self, file_path: str) -> Dict:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def get_audio_properties(self, file_path: str) -> Dict:
        try:
            data = self._ffprobe(file_path)
            audio_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None
            )
            if not audio_stream:
                return {}
            duration = float(data.get('format', {}).get('duration', 0))
            channels = int(audio_stream.get('channels', 0))
            if self.audio_quality:
                preset = QualityPreset.audio(self.audio_quality)
                bit_depth = preset['bit_depth']
            else:
                bit_depth = int(audio_stream.get('bits_per_sample', 0)) or None
            return {
                'sample_rate_hz':     int(audio_stream.get('sample_rate', 0)),
                'bit_depth':          bit_depth,
                'channels':           channels,
                'is_mono':            channels == 1,
                'codec':              audio_stream.get('codec_name', ''),
                'bitrate_kbps':       int(audio_stream.get('bit_rate', 0)) // 1000,
                'audio_duration_sec': round(duration, 2),
                'audio_duration_hms': fmt_duration(duration),
            }
        except Exception as e:
            print(f"⚠️  Could not extract audio properties: {e}")
            return {}

    def get_video_properties(self, file_path: str) -> Dict:
        try:
            data = self._ffprobe(file_path)
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None
            )
            if not video_stream:
                return {}
            width  = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            fps_raw = video_stream.get('r_frame_rate', '0/1')
            num, den = (int(x) for x in fps_raw.split('/'))
            fps = round(num / den, 3) if den else 0
            return {
                'codec':      video_stream.get('codec_name', ''),
                'width':      width,
                'height':     height,
                'resolution': f"{width}x{height}",
                'fps':        fps,
                'bitrate_kbps': int(video_stream.get('bit_rate', 0)) // 1000,
            }
        except Exception as e:
            print(f"⚠️  Could not extract video properties: {e}")
            return {}

    def process_video(self, video_id: str, url: str, source: str = '') -> bool:
        print(f"\n📹 Processing: {video_id}")

        if self.video_exists(video_id):
            print(f"⏭️  Already downloaded, skipping")
            return True

        info = self.get_video_info(video_id)
        if not info:
            print(f"❌ Could not get video info")
            return False

        if video_id not in self.data:
            self.data[video_id] = self._build_entry(url, info, source)
            self._save()

        if self.audio_quality:
            print(f"⬇️  Downloading audio ({self.audio_quality})...")
            abs_path = self.download_audio(video_id)
            if not abs_path:
                self.mark_failed(video_id, "Audio download failed")
                return False
            audio_props = self.get_audio_properties(abs_path)
            p = Path(abs_path)
            file_size = p.stat().st_size if p.exists() else None
            rel_path = str(p.relative_to(self.output_dir.parent))
        else:
            print(f"⬇️  Downloading video ({self.video_quality})...")
            abs_path = self.download_video(video_id)
            if not abs_path:
                self.mark_failed(video_id, "Video download failed")
                return False
            audio_props = self.get_audio_properties(abs_path)
            video_props = self.get_video_properties(abs_path)
            p = Path(abs_path)
            file_size = p.stat().st_size if p.exists() else None
            rel_path = str(p.relative_to(self.output_dir.parent))

        print(f"📝 Downloading subtitles...")
        subtitle_files = self.download_subtitles(video_id, info)

        self.mark_completed(video_id, rel_path, file_size, audio_props,
                            video_props=video_props if self.video_quality else None)
        if subtitle_files:
            e = self.data.get(video_id)
            if e:
                e['subtitle_files'] = subtitle_files
                self._save()
        print(f"✅ Completed: {video_id}")
        return True

    def process_url(self, url: str):
        self.add_source(url)
        print(f"\n🔍 Extracting video IDs from URL...")
        video_ids = self.extract_video_ids(url)

        if not video_ids:
            print("❌ No videos found")
            return

        print(f"📊 Found {len(video_ids)} video(s)")
        successful = failed = skipped = 0

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}]", end=" ")
            if self.video_exists(video_id):
                print(f"⏭️  {video_id} - Already downloaded")
                skipped += 1
                continue
            if self.process_video(video_id, f"https://www.youtube.com/watch?v={video_id}", source=url):
                successful += 1
            else:
                failed += 1

        self.refresh_stats()
        self._save()

        print(f"\n{'='*60}")
        print(f"📊 Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   📁 Output: {self.output_dir}")
        print(f"{'='*60}\n")


def cmd_check(dataset_dir: Path):
    metadata_path = dataset_dir / 'getData.json'
    if not metadata_path.exists():
        print(f"❌ getData.json not found in {dataset_dir}")
        sys.exit(1)

    raw = json.loads(metadata_path.read_text(encoding='utf-8'))
    meta = raw.pop('_meta', {})
    data = raw
    base_dir = dataset_dir.parent

    stats = meta
    dur   = meta.get('duration_stats', {})
    print(f"\n{'='*60}")
    print(f"📦 {meta.get('dataset_name', dataset_dir.name)}")
    print(f"   Created      : {meta.get('created_at', '')[:10]}")
    print(f"   Last updated : {meta.get('last_updated_at', '')[:10]}")
    print(f"   Total        : {stats.get('total', 0)}  |  ✅ {stats.get('completed', 0)}  ❌ {stats.get('failed', 0)}  ⏳ {stats.get('pending', 0)}")
    print(f"   Audio total  : {stats.get('total_audio_duration_hms', '—')}  ({stats.get('total_audio_duration_sec', 0):.0f}s)")
    print(f"   Size         : {stats.get('total_file_size_mb', 0):.1f} MB")
    if dur.get('avg_hms'):
        print(f"   Duration     : avg {dur.get('avg_hms')}  min {dur.get('min_hms')}  max {dur.get('max_hms')}")
    print(f"{'='*60}\n")

    completed = {vid: e for vid, e in data.items() if e.get('status') == 'completed'}
    if not completed:
        print("No completed entries to check.")
        return

    print(f"🔍 Checking files for {len(completed)} completed entries...\n")
    missing = []

    for video_id, entry in completed.items():
        title_short = (entry.get('title') or video_id)[:55]
        problems = []

        fp = entry.get('file_path')
        if fp and not (base_dir / fp).exists():
            problems.append(f"file missing: {fp}")

        for lang, sp in (entry.get('subtitle_files') or {}).items():
            if not (base_dir / sp).exists():
                problems.append(f"subtitle missing: {sp}")

        if problems:
            missing.append((video_id, title_short, problems))
            print(f"❌ {video_id}  {title_short}")
            for p in problems:
                print(f"     {p}")
        else:
            print(f"✅ {video_id}  {title_short}")

    print(f"\n{'='*60}")
    print(f"✅ OK: {len(completed) - len(missing)}   ❌ Missing files: {len(missing)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='YouTube Audio/Video Dataset Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--nodejs-path', help='Custom Node.js installation path')
    subparsers = parser.add_subparsers(dest='command', required=True)

    dl = subparsers.add_parser(
        'download',
        help='Download audio or video from a YouTube URL',
        epilog="""
Examples:
  %(prog)s https://www.youtube.com/@Channel -d my_dataset -a best
  %(prog)s https://www.youtube.com/@Channel -d my_dataset -v eco
  %(prog)s https://www.youtube.com/watch?v=ID -d single -a eco
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dl.add_argument('url', help='YouTube URL (video, playlist, or channel)')
    dl.add_argument('-d', '--dataset', required=True, help='Dataset name')
    mode = dl.add_mutually_exclusive_group(required=True)
    mode.add_argument('-a', '--audio', choices=['best', 'high', 'medium', 'low'], metavar='PRESET',
                      help='Download audio — best (44k) high (24k) medium (22.5k) low (16k)')
    mode.add_argument('-v', '--video', choices=['max', 'best', 'high', 'medium', 'low'], metavar='PRESET',
                      help='Download video — max (no limit) best (1080p) high (720p) medium (480p) low (360p)')

    ck = subparsers.add_parser(
        'check',
        help='Verify that files recorded in metadata actually exist on disk',
        epilog="Example:\n  %(prog)s -d my_dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ck.add_argument('-d', '--dataset', required=True, help='Dataset name')

    args = parser.parse_args()

    if args.nodejs_path:
        nodejs_path = Path(args.nodejs_path)
        if nodejs_path.exists():
            os.environ['PATH'] = f"{nodejs_path}{os.pathsep}{os.environ['PATH']}"
            print(f"✅ Added Node.js path: {nodejs_path}")
        else:
            print(f"⚠️  Warning: Node.js path not found: {nodejs_path}")

    output_dir = Path('downloads') / args.dataset

    if args.command == 'check':
        try:
            cmd_check(output_dir)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
            sys.exit(1)
        return

    print(f"\n{'='*60}")
    print(f"🎵 YouTube Dataset Downloader")
    print(f"{'='*60}")
    print(f"Dataset : {args.dataset}")
    if args.audio:
        p = QualityPreset.audio(args.audio)
        print(f"Mode    : audio ({args.audio}) — {p['sample_rate']} Hz, {p['bit_depth']}-bit, mono")
    else:
        p = QualityPreset.video(args.video)
        print(f"Mode    : video ({args.video}) — {p['description']} max")
    print(f"Output  : {output_dir}")
    print(f"{'='*60}\n")

    downloader = YouTubeDownloader(
        dataset_name=args.dataset,
        output_dir=str(output_dir),
        audio_quality=args.audio,
        video_quality=args.video,
    )

    try:
        downloader.process_url(args.url)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
