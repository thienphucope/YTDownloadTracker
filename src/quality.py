from typing import Dict


class QualityPreset:
    """Quality presets for audio and video downloads."""

    AUDIO = {
        'best': {
            'sample_rate': 24000,
            'bit_depth': 16,
            'channels': 1,
            'description': 'High-quality audio for training',
        },
        'eco': {
            'sample_rate': 16000,
            'bit_depth': 16,
            'channels': 1,
            'description': 'Storage-efficient audio',
        },
    }

    VIDEO = {
        'best': {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
            'description': 'Best available video quality',
        },
        'eco': {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]',
            'description': '720p video',
        },
    }

    @classmethod
    def audio(cls, name: str) -> Dict:
        return cls.AUDIO.get(name, cls.AUDIO['best'])

    @classmethod
    def video(cls, name: str) -> Dict:
        return cls.VIDEO.get(name, cls.VIDEO['best'])
