#!/usr/bin/env python3
"""
YouTube Audio/Video Dataset Downloader
Downloads audio or video from YouTube URLs and tracks metadata for dataset management.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yt_dlp
import subprocess
import re

from src.quality import QualityPreset
from src.metadata import MetadataManager


class YouTubeDownloader:
    """Downloads audio or video from YouTube and manages dataset."""

    def __init__(self, dataset_name: str, output_dir: str,
                 audio_quality: Optional[str] = None, video_quality: Optional[str] = None):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.audio_quality = audio_quality
        self.video_quality = video_quality

        self.subtitles_dir = self.output_dir / 'subtitles'
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)

        if audio_quality:
            self.wavs_dir = self.output_dir / 'wavs'
            self.wavs_dir.mkdir(parents=True, exist_ok=True)
        if video_quality:
            self.videos_dir = self.output_dir / 'videos'
            self.videos_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = MetadataManager(
            self.output_dir / 'metadata.json',
            dataset_name=dataset_name,
            audio_preset=audio_quality,
            video_preset=video_quality,
        )

    def extract_video_ids(self, url: str) -> List[str]:
        """Extract video IDs from URL (handles single video, playlist, channel)."""
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
        """Get video information."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
                return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        except Exception as e:
            print(f"❌ Error getting video info for {video_id}: {e}")
            return None

    def download_audio(self, video_id: str) -> Optional[str]:
        """Download and convert audio for a video."""
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
        """Convert audio to target quality using ffmpeg."""
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
        """Download video file."""
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
        """Download subtitles with timestamps.

        Manual subs: video's primary language only.
        Auto captions: en and vi only.
        """
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

    def get_audio_properties(self, file_path: str) -> Dict:
        """Extract audio properties using ffprobe."""
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
            preset = QualityPreset.audio(self.audio_quality)
            duration = float(data.get('format', {}).get('duration', 0))
            return {
                'sample_rate_hz': int(audio_stream.get('sample_rate', 0)),
                'bit_depth': preset['bit_depth'],
                'channels': int(audio_stream.get('channels', 0)),
                'is_mono': int(audio_stream.get('channels', 0)) == 1,
                'codec': audio_stream.get('codec_name', ''),
                'bitrate_kbps': int(audio_stream.get('bit_rate', 0)) // 1000,
                'audio_duration_sec': round(duration, 2),
            }
        except Exception as e:
            print(f"⚠️  Could not extract audio properties: {e}")
            return {}

    def process_video(self, video_id: str, url: str) -> bool:
        """Process a single video: download audio or video."""
        print(f"\n📹 Processing: {video_id}")

        if self.metadata.video_exists(video_id):
            print(f"⏭️  Already downloaded, skipping")
            return True

        info = self.get_video_info(video_id)
        if not info:
            print(f"❌ Could not get video info")
            return False

        if video_id not in self.metadata.data:
            self.metadata.create_entry(video_id, url, info)

        if self.audio_quality:
            print(f"⬇️  Downloading audio ({self.audio_quality})...")
            abs_path = self.download_audio(video_id)
            if not abs_path:
                self.metadata.mark_failed(video_id, "Audio download failed")
                return False
            audio_props = self.get_audio_properties(abs_path)
            p = Path(abs_path)
            file_size = p.stat().st_size if p.exists() else None
            rel_path = str(p.relative_to(self.output_dir.parent))

        else:
            print(f"⬇️  Downloading video ({self.video_quality})...")
            abs_path = self.download_video(video_id)
            if not abs_path:
                self.metadata.mark_failed(video_id, "Video download failed")
                return False
            audio_props = {}
            p = Path(abs_path)
            file_size = p.stat().st_size if p.exists() else None
            rel_path = str(p.relative_to(self.output_dir.parent))

        print(f"📝 Downloading subtitles...")
        subtitle_files = self.download_subtitles(video_id, info)

        self.metadata.mark_completed(video_id, rel_path, file_size, audio_props)
        if subtitle_files:
            self.metadata.update_entry(video_id, subtitle_files=subtitle_files)
        print(f"✅ Completed: {video_id}")
        return True

    def process_url(self, url: str):
        """Process URL (single video, playlist, or channel)."""
        self.metadata.add_source(url)
        print(f"\n🔍 Extracting video IDs from URL...")
        video_ids = self.extract_video_ids(url)

        if not video_ids:
            print("❌ No videos found")
            return

        print(f"📊 Found {len(video_ids)} video(s)")
        successful = failed = skipped = 0

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}]", end=" ")
            if self.metadata.video_exists(video_id):
                print(f"⏭️  {video_id} - Already downloaded")
                skipped += 1
                continue
            if self.process_video(video_id, f"https://www.youtube.com/watch?v={video_id}"):
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"📊 Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   📁 Output: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='YouTube Audio/Video Dataset Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download audio
  %(prog)s https://www.youtube.com/@Channel -d my_dataset -a best

  # Download video
  %(prog)s https://www.youtube.com/@Channel -d my_dataset -v eco

  # Single video, eco audio
  %(prog)s https://www.youtube.com/watch?v=... -d single -a eco
        """
    )

    parser.add_argument('url', help='YouTube URL (video, playlist, or channel)')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset name')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-a', '--audio', choices=['best', 'eco'], metavar='PRESET',
                      help='Download audio. Presets: best (24kHz), eco (16kHz)')
    mode.add_argument('-v', '--video', choices=['best', 'eco'], metavar='PRESET',
                      help='Download video. Presets: best (max quality), eco (720p)')

    parser.add_argument('--nodejs-path', help='Custom Node.js installation path')

    args = parser.parse_args()

    output_dir = Path('downloads') / args.dataset

    if args.nodejs_path:
        nodejs_path = Path(args.nodejs_path)
        if nodejs_path.exists():
            os.environ['PATH'] = f"{nodejs_path}{os.pathsep}{os.environ['PATH']}"
            print(f"✅ Added Node.js path: {nodejs_path}")
        else:
            print(f"⚠️  Warning: Node.js path not found: {nodejs_path}")

    print(f"\n{'='*60}")
    print(f"🎵 YouTube Dataset Downloader")
    print(f"{'='*60}")
    print(f"Dataset : {args.dataset}")
    if args.audio:
        p = QualityPreset.audio(args.audio)
        print(f"Mode    : audio ({args.audio}) — {p['sample_rate']} Hz, {p['bit_depth']}-bit, mono")
    else:
        p = QualityPreset.video(args.video)
        print(f"Mode    : video ({args.video}) — {p['description']}")
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
