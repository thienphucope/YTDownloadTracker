from typing import Dict


class QualityPreset:

    AUDIO = {
        'best':   {'sample_rate': 44100, 'bit_depth': 16, 'channels': 1},
        'high':   {'sample_rate': 24000, 'bit_depth': 16, 'channels': 1},
        'medium': {'sample_rate': 22050, 'bit_depth': 16, 'channels': 1},
        'low':    {'sample_rate': 16000, 'bit_depth': 16, 'channels': 1},
    }

    VIDEO = {
        'max':    {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',                                         'description': 'max available'},
        'best':   {'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]', 'description': '1080p'},
        'high':   {'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]',   'description': '720p'},
        'medium': {'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]',   'description': '480p'},
        'low':    {'format': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360]+bestaudio/best[height<=360]',   'description': '360p'},
    }

    @classmethod
    def audio(cls, name: str) -> Dict:
        return cls.AUDIO.get(name, cls.AUDIO['high'])

    @classmethod
    def video(cls, name: str) -> Dict:
        return cls.VIDEO.get(name, cls.VIDEO['high'])
