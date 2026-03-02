#!/usr/bin/env python3
"""
YouTube Audio Dataset Downloader
Downloads audio from YouTube URLs and tracks metadata for dataset management.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yt_dlp
import subprocess
import re


class AudioQualityPreset:
    """Audio quality presets for downloads."""
    
    PRESETS = {
        'best': {
            'format': 'wav',
            'sample_rate': 24000,
            'bit_depth': 16,
            'channels': 1,
            'description': 'High-quality audio for training'
        },
        'eco': {
            'format': 'wav',
            'sample_rate': 16000,
            'bit_depth': 16,
            'channels': 1,
            'description': 'Storage-efficient quality'
        }
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Dict:
        """Get audio quality preset by name."""
        return cls.PRESETS.get(name, cls.PRESETS['best'])


class MetadataManager:
    """Manages metadata.json for tracking downloads."""
    
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save(self):
        """Save metadata to file."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def video_exists(self, video_id: str) -> bool:
        """Check if video already downloaded."""
        entry = self.data.get(video_id, {})
        return entry.get('status') == 'completed'
    
    def create_entry(self, video_id: str, url: str, info: Dict):
        """Create metadata entry for new video."""
        self.data[video_id] = {
            'video_id': video_id,
            'url': url,
            'title': info.get('title', ''),
            'channel': info.get('uploader', ''),
            'description': info.get('description', '')[:500] if info.get('description') else '',
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'file_path': None,
            'file_size_bytes': None,
            'audio_properties': {},
            'retry_count': 0,
            'error': None
        }
        self.save()
    
    def update_entry(self, video_id: str, **kwargs):
        """Update metadata entry."""
        if video_id in self.data:
            self.data[video_id].update(kwargs)
            self.data[video_id]['updated_at'] = datetime.now().isoformat()
            self.save()
    
    def mark_completed(self, video_id: str, file_path: str, audio_props: Dict):
        """Mark download as completed."""
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
        self.update_entry(
            video_id,
            status='completed',
            file_path=str(file_path),
            file_size_bytes=file_size,
            audio_properties=audio_props
        )
    
    def mark_failed(self, video_id: str, error: str):
        """Mark download as failed."""
        entry = self.data.get(video_id, {})
        retry_count = entry.get('retry_count', 0) + 1
        self.update_entry(
            video_id,
            status='failed',
            error=str(error),
            retry_count=retry_count
        )


class YouTubeAudioDownloader:
    """Downloads audio from YouTube and manages dataset."""
    
    def __init__(self, dataset_name: str, output_dir: str, quality: str = 'best'):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.quality_preset = AudioQualityPreset.get_preset(quality)
        
        # Setup directories
        self.wavs_dir = self.output_dir / 'wavs'
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup metadata
        self.metadata = MetadataManager(self.output_dir / 'metadata.json')
    
    def extract_video_ids(self, url: str) -> List[str]:
        """Extract video IDs from URL (handles single video, playlist, channel)."""
        # Fix channel URLs to point to videos tab
        if '@' in url or '/c/' in url or '/channel/' in url or '/user/' in url:
            if not url.endswith('/videos'):
                url = url.rstrip('/') + '/videos'
        
        ydl_opts = {
            'quiet': False,
            'extract_flat': 'in_playlist',
            'skip_download': True,
            'ignoreerrors': True,
        }
        
        video_ids = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    print(f"❌ Could not extract info from URL")
                    return []
                
                if 'entries' in info:
                    # Playlist or channel
                    for entry in info['entries']:
                        if entry and 'id' in entry:
                            # Skip if ID looks like a channel/user ID
                            if not entry['id'].startswith('UC') or len(entry['id']) != 24:
                                video_ids.append(entry['id'])
                            elif 'url' in entry:
                                # Extract video ID from URL
                                match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&/]|$)', entry['url'])
                                if match:
                                    video_ids.append(match.group(1))
                else:
                    # Single video
                    if 'id' in info:
                        video_ids.append(info['id'])
        
        except Exception as e:
            print(f"❌ Error extracting video IDs: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen and len(vid) == 11:  # YouTube video IDs are 11 chars
                seen.add(vid)
                unique_ids.append(vid)
        
        return unique_ids
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get video information."""
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://www.youtube.com/watch?v={video_id}"
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            print(f"❌ Error getting video info for {video_id}: {e}")
            return None
    
    def download_audio(self, video_id: str) -> Optional[str]:
        """Download audio for a video."""
        output_path = self.wavs_dir / f"{video_id}.wav"
        temp_path = self.wavs_dir / f"{video_id}_temp.wav"
        converted_path = self.wavs_dir / f"{video_id}_converted.wav"
        
        # yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(temp_path.with_suffix('.%(ext)s')),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': False,
            'no_warnings': False,
        }
        
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if downloaded file exists
            if not temp_path.exists():
                raise FileNotFoundError(f"Downloaded file not found: {temp_path}")
            
            print(f"📁 Downloaded: {temp_path.name} ({temp_path.stat().st_size / 1024:.1f} KB)")
            
            # Convert to target quality using ffmpeg
            print(f"🔄 Converting to {self.quality_preset['sample_rate']}Hz mono...")
            self._convert_audio(temp_path, converted_path)
            
            if not converted_path.exists():
                raise FileNotFoundError(f"Converted file not found: {converted_path}")
            
            print(f"📁 Converted: {converted_path.name} ({converted_path.stat().st_size / 1024:.1f} KB)")
            
            # Move converted file to final location
            if output_path.exists():
                print(f"⚠️  Removing existing: {output_path.name}")
                output_path.unlink()
            
            converted_path.rename(output_path)
            print(f"📁 Final: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")
            
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()
                print(f"🗑️  Cleaned up: {temp_path.name}")
            
            # Verify final file exists
            if not output_path.exists():
                raise FileNotFoundError(f"Final file missing: {output_path}")
            
            return str(output_path)
        
        except Exception as e:
            print(f"❌ Download failed for {video_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup temp files
            for pattern in [f"{video_id}_temp.*", f"{video_id}_converted.*"]:
                for f in self.wavs_dir.glob(pattern):
                    try:
                        print(f"🗑️  Cleaning up: {f.name}")
                        f.unlink()
                    except Exception as cleanup_err:
                        print(f"⚠️  Cleanup failed for {f.name}: {cleanup_err}")
            return None
    
    def _convert_audio(self, input_path: Path, output_path: Path):
        """Convert audio to target quality using ffmpeg."""
        preset = self.quality_preset
        
        # Ensure input and output are different
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must be different")
        
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ar', str(preset['sample_rate']),
            '-ac', str(preset['channels']),
            '-sample_fmt', f's{preset["bit_depth"]}',
            '-n',  # Never overwrite
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
    
    def get_audio_properties(self, file_path: str) -> Dict:
        """Extract audio properties using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                return {}
            
            duration = float(data.get('format', {}).get('duration', 0))
            
            return {
                'sample_rate_hz': int(audio_stream.get('sample_rate', 0)),
                'bit_depth': self.quality_preset['bit_depth'],
                'channels': int(audio_stream.get('channels', 0)),
                'is_mono': int(audio_stream.get('channels', 0)) == 1,
                'codec': audio_stream.get('codec_name', ''),
                'bitrate_kbps': int(audio_stream.get('bit_rate', 0)) // 1000,
                'audio_duration_sec': round(duration, 2)
            }
        
        except Exception as e:
            print(f"⚠️  Could not extract audio properties: {e}")
            return {}
    
    def process_video(self, video_id: str, url: str) -> bool:
        """Process a single video download."""
        print(f"\n📹 Processing: {video_id}")
        
        # Check if already downloaded
        if self.metadata.video_exists(video_id):
            print(f"⏭️  Already downloaded, skipping")
            return True
        
        # Get video info
        info = self.get_video_info(video_id)
        if not info:
            print(f"❌ Could not get video info")
            return False
        
        # Create metadata entry
        if video_id not in self.metadata.data:
            self.metadata.create_entry(video_id, url, info)
        
        # Download audio
        print(f"⬇️  Downloading audio...")
        output_path = self.download_audio(video_id)
        if not output_path:
            self.metadata.mark_failed(video_id, "Download failed")
            return False
        
        # Get audio properties
        audio_props = self.get_audio_properties(output_path)
        
        # Mark completed
        self.metadata.mark_completed(video_id, output_path, audio_props)
        print(f"✅ Completed: {Path(output_path).name}")
        
        return True
    
    def process_url(self, url: str):
        """Process URL (single video, playlist, or channel)."""
        print(f"\n🔍 Extracting video IDs from URL...")
        video_ids = self.extract_video_ids(url)
        
        if not video_ids:
            print("❌ No videos found")
            return
        
        print(f"📊 Found {len(video_ids)} video(s)")
        
        # Process each video
        successful = 0
        failed = 0
        skipped = 0
        
        for i, video_id in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}]", end=" ")
            
            if self.metadata.video_exists(video_id):
                print(f"⏭️  {video_id} - Already downloaded")
                skipped += 1
                continue
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            if self.process_video(video_id, video_url):
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   📁 Output: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='YouTube Audio Dataset Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from channel
  %(prog)s https://www.youtube.com/@SolusAstorias -d solus -q best

  # Download from playlist
  %(prog)s https://www.youtube.com/playlist?list=... -d my_dataset -q eco

  # Download single video
  %(prog)s https://www.youtube.com/watch?v=... -d single -q best
        """
    )
    
    parser.add_argument('url', help='YouTube URL (video, playlist, or channel)')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset name')
    parser.add_argument('-q', '--quality', choices=['best', 'eco'], default='best',
                       help='Audio quality preset (default: best)')
    parser.add_argument('--nodejs-path', help='Custom Node.js installation path (e.g., D:\\Programs\\nodejs)')
    
    args = parser.parse_args()

    # Set default output directory if not provided
    output_dir = Path('downloads') / args.dataset

    # Add custom Node.js path to environment if provided
    if args.nodejs_path:
        nodejs_path = Path(args.nodejs_path)
        if nodejs_path.exists():
            os.environ['PATH'] = f"{nodejs_path}{os.pathsep}{os.environ['PATH']}"
            print(f"✅ Added Node.js path: {nodejs_path}")
        else:
            print(f"⚠️  Warning: Node.js path not found: {nodejs_path}")

    # Print configuration
    preset = AudioQualityPreset.get_preset(args.quality)
    print(f"\n{'='*60}")
    print(f"🎵 YouTube Audio Dataset Downloader")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Quality: {args.quality} ({preset['description']})")
    print(f"  - Sample Rate: {preset['sample_rate']} Hz")
    print(f"  - Bit Depth: {preset['bit_depth']}-bit")
    print(f"  - Channels: {preset['channels']} (Mono)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Initialize downloader
    downloader = YouTubeAudioDownloader(
        dataset_name=args.dataset,
        output_dir=str(output_dir),
        quality=args.quality
    )
    
    # Process URL
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