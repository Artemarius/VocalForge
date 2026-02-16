"""Audio file I/O — load, save, and inspect audio files.

Wraps soundfile for consistent float32 numpy array handling.
"""

import numpy as np
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 numpy array.

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of (data, sample_rate). Data shape is (samples,) for mono
        or (samples, channels) for multi-channel.

    Raises:
        FileNotFoundError: If the file does not exist.
        soundfile.SoundFileError: If the file cannot be read.
        ValueError: If the file contains no audio data.
    """
    data, sample_rate = sf.read(path, dtype="float32")

    if data.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    return data, sample_rate


def save_audio(path: str, data: np.ndarray, sample_rate: int) -> None:
    """Save a numpy array as an audio file.

    Args:
        path: Output file path. Format is inferred from extension.
        data: Audio data as float32 numpy array.
        sample_rate: Sample rate in Hz.
    """
    sf.write(path, data, sample_rate)


def get_audio_info(path: str) -> dict:
    """Get metadata about an audio file without loading it.

    Args:
        path: Path to the audio file.

    Returns:
        Dict with keys: frames, channels, sample_rate, duration, format.
    """
    info = sf.info(path)
    return {
        "frames": info.frames,
        "channels": info.channels,
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "format": info.format,
    }
