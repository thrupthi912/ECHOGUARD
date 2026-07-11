"""
echoguard.utils
~~~~~~~~~~~~~~~
Shared utility helpers used across all EchoGuard modules.

Imports are lazy to avoid pulling in numpy/librosa at module import time,
which can break on environments where packages are in user site-packages.
"""


def __getattr__(name):
    if name in ("load_audio", "save_audio", "get_duration"):
        from echoguard.utils.audio import load_audio, save_audio, get_duration
        globals().update({"load_audio": load_audio, "save_audio": save_audio,
                          "get_duration": get_duration})
        return globals()[name]
    if name in ("sanitize_text", "format_diarized_transcript"):
        from echoguard.utils.text import sanitize_text, format_diarized_transcript
        globals().update({"sanitize_text": sanitize_text,
                          "format_diarized_transcript": format_diarized_transcript})
        return globals()[name]
    if name == "load_config":
        from echoguard.utils.config import load_config
        globals()["load_config"] = load_config
        return load_config
    raise AttributeError(f"module 'echoguard.utils' has no attribute {name!r}")


__all__ = [
    "load_audio",
    "save_audio",
    "get_duration",
    "sanitize_text",
    "format_diarized_transcript",
    "load_config",
]
