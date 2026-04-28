import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


class MetadataManager:
    """
    Manages metadata.json for a dataset.

    JSON structure (nested for readability):
      {
        "_meta": { ... },
        "<video_id>": {
          identity fields...,
          "yt_info":  { youtube metadata },
          "download": { preset, status, timestamps, retry },
          "files":    { path, size, audio_properties, subtitles }
        }
      }

    Internally entries are kept flat for simple access.
    _to_json_entry / _from_json_entry handle reshaping on save/load.
    Schema is intentionally open — new fields added via update_entry() are
    preserved as-is at the top level.
    """

    # Fields that belong to each JSON group
    _YT_INFO_KEYS = {
        'video_id', 'url', 'title', 'channel', 'channel_id', 'upload_date',
        'availability', 'live_status', 'age_limit', 'duration', 'duration_hms',
        'view_count', 'like_count', 'channel_follower_count', 'description',
    }
    _DOWNLOAD_KEYS = {
        'audio_preset', 'video_preset', 'status', 'count',
        'created_at', 'updated_at', 'last_checked_at', 'retry_count', 'error',
    }
    _FILES_KEYS = {
        'file_path', 'file_size_bytes', 'audio_properties', 'subtitle_files',
    }

    def __init__(self, metadata_path: Path, dataset_name: str = '',
                 audio_preset: Optional[str] = None, video_preset: Optional[str] = None):
        self.metadata_path = metadata_path
        self.data, self.meta = self._load()
        self._initialize_meta(dataset_name, audio_preset, video_preset)
        self._ensure_counts()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self) -> Tuple[Dict, Dict]:
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            meta = raw.pop('_meta', {})
            data = {vid: self._from_json_entry(entry) for vid, entry in raw.items()}
            return data, meta
        return {}, {}

    def save(self):
        self._update_stats()
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        out = {'_meta': self.meta}
        out.update({vid: self._to_json_entry(entry) for vid, entry in self.data.items()})
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Entry reshaping
    # ------------------------------------------------------------------

    def _to_json_entry(self, e: Dict) -> Dict:
        """Reshape flat entry into grouped JSON structure."""
        known = self._YT_INFO_KEYS | self._DOWNLOAD_KEYS | self._FILES_KEYS
        extra = {k: v for k, v in e.items() if k not in known}

        return {
            'yt_info': {
                'video_id':               e.get('video_id'),
                'url':                    e.get('url'),
                'title':                  e.get('title'),
                'channel':                e.get('channel'),
                'channel_id':             e.get('channel_id'),
                'upload_date':            e.get('upload_date'),
                'availability':           e.get('availability'),
                'live_status':            e.get('live_status'),
                'age_limit':              e.get('age_limit'),
                'duration':               e.get('duration'),
                'duration_hms':           e.get('duration_hms'),
                'view_count':             e.get('view_count'),
                'like_count':             e.get('like_count'),
                'channel_follower_count': e.get('channel_follower_count'),
                'description':            e.get('description'),
            },
            'download': {
                'audio_preset':    e.get('audio_preset'),
                'video_preset':    e.get('video_preset'),
                'status':          e.get('status'),
                'count':           e.get('count'),
                'created_at':      e.get('created_at'),
                'updated_at':      e.get('updated_at'),
                'last_checked_at': e.get('last_checked_at'),
                'retry_count':     e.get('retry_count'),
                'error':           e.get('error'),
            },
            'files': {
                'file_path':       e.get('file_path'),
                'file_size_bytes': e.get('file_size_bytes'),
                'file_size_mb':    round(e['file_size_bytes'] / (1024 * 1024), 2) if e.get('file_size_bytes') else None,
                'audio_properties': {
                    **{k: v for k, v in e.get('audio_properties', {}).items() if k != 'audio_duration_hms'},
                    'audio_duration_hms': self._fmt_duration(
                        e.get('audio_properties', {}).get('audio_duration_sec')
                    ),
                } if e.get('audio_properties') else {},
                'subtitle_files':  e.get('subtitle_files', {}),
            },
            **extra,  # future pipeline fields land here
        }

    def _from_json_entry(self, e: Dict) -> Dict:
        """Flatten grouped JSON entry back to a flat dict."""
        if 'yt_info' not in e and 'download' not in e:
            return e  # legacy flat format
        flat = {k: v for k, v in e.items() if k not in ('yt_info', 'download', 'files')}
        flat.update(e.get('yt_info', {}))
        flat.update(e.get('download', {}))
        flat.update(e.get('files', {}))
        return flat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_meta(self, dataset_name: str,
                         audio_preset: Optional[str], video_preset: Optional[str]):
        is_new = not self.meta
        self.meta.setdefault('dataset_name', dataset_name)
        self.meta.setdefault('audio_preset', audio_preset)
        self.meta.setdefault('video_preset', video_preset)
        self.meta.setdefault('sources', [])
        # timestamps grouped together, stats last
        self.meta.setdefault('created_at', datetime.now().isoformat())
        self.meta.setdefault('last_updated_at', datetime.now().isoformat())
        self.meta.setdefault('stats', {})
        if is_new and self.metadata_path.exists():
            self.save()

    @staticmethod
    def _fmt_duration(secs) -> Optional[str]:
        if secs is None:
            return None
        secs = int(secs)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_stats(self):
        completed = [e for e in self.data.values() if e.get('status') == 'completed']
        pending_ids = [vid for vid, e in self.data.items() if e.get('status') == 'pending']
        failed_ids  = [vid for vid, e in self.data.items() if e.get('status') == 'failed']

        durations = [e['duration'] for e in completed if e.get('duration')]
        dur_stats = {
            'max_sec': max(durations) if durations else None,
            'min_sec': min(durations) if durations else None,
            'avg_sec': round(sum(durations) / len(durations), 1) if durations else None,
            'max_hms': self._fmt_duration(max(durations)) if durations else None,
            'min_hms': self._fmt_duration(min(durations)) if durations else None,
            'avg_hms': self._fmt_duration(sum(durations) / len(durations)) if durations else None,
        }

        total_audio_sec = round(
            sum(e.get('audio_properties', {}).get('audio_duration_sec', 0) for e in completed), 2
        )
        total_size_bytes = sum(e.get('file_size_bytes', 0) or 0 for e in completed)

        self.meta['last_updated_at'] = datetime.now().isoformat()
        self.meta['stats'] = {
            'total':     len(self.data),
            'completed': len(completed),
            'pending':   len(pending_ids),
            'pending_ids': pending_ids,
            'failed':    len(failed_ids),
            'failed_ids': failed_ids,
            'total_audio_duration_sec': total_audio_sec,
            'total_audio_duration_hms': self._fmt_duration(total_audio_sec),
            'total_file_size_bytes': total_size_bytes,
            'total_file_size_mb':    round(total_size_bytes / (1024 * 1024), 2),
            'duration_stats': dur_stats,
        }

    def _ensure_counts(self):
        changed = False
        for idx, (vid, entry) in enumerate(self.data.items(), start=1):
            if 'count' not in entry:
                entry['count'] = idx
                changed = True
        if changed:
            self.save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_source(self, url: str):
        if url not in self.meta.setdefault('sources', []):
            self.meta['sources'].append(url)
            self.save()

    def video_exists(self, video_id: str) -> bool:
        return self.data.get(video_id, {}).get('status') == 'completed'

    def create_entry(self, video_id: str, url: str, info: Dict):
        upload_date = info.get('upload_date', '')
        if upload_date and len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

        now = datetime.now().isoformat()
        self.data[video_id] = {
            # identity
            'video_id':   video_id,
            'url':        url,
            'title':      info.get('title', ''),
            'channel':    info.get('uploader', ''),
            'channel_id': info.get('channel_id', ''),
            'upload_date': upload_date,
            # yt_info
            'availability':           info.get('availability', 'public'),
            'live_status':            info.get('live_status', 'not_live'),
            'age_limit':              info.get('age_limit', 0),
            'duration':               info.get('duration'),
            'duration_hms':           self._fmt_duration(info.get('duration')),
            'view_count':             info.get('view_count'),
            'like_count':             info.get('like_count'),
            'channel_follower_count': info.get('channel_follower_count'),
            'description':            info.get('description', '')[:500] if info.get('description') else '',
            # download
            'audio_preset':    self.meta.get('audio_preset'),
            'video_preset':    self.meta.get('video_preset'),
            'status':          'pending',
            'count':           len(self.data) + 1,
            'created_at':      now,
            'updated_at':      now,
            'last_checked_at': now,
            'retry_count':     0,
            'error':           None,
            # files
            'file_path':        None,
            'file_size_bytes':  None,
            'audio_properties': {},
            'subtitle_files':   {},
        }
        self.save()

    def update_entry(self, video_id: str, **kwargs):
        if video_id in self.data:
            self.data[video_id].update(kwargs)
            self.data[video_id]['updated_at'] = datetime.now().isoformat()
            self.save()

    def mark_completed(self, video_id: str, file_path: str,
                       file_size: Optional[int], audio_props: Dict):
        self.update_entry(
            video_id,
            status='completed',
            file_path=file_path,
            file_size_bytes=file_size,
            audio_properties=audio_props,
        )

    def mark_failed(self, video_id: str, error: str):
        retry_count = self.data.get(video_id, {}).get('retry_count', 0) + 1
        self.update_entry(video_id, status='failed', error=str(error), retry_count=retry_count)
