"""Demucs source separation — extract instrumental from a song.

Pure Python module (no PySide6). Runs Demucs htdemucs_ft model to separate
a song into stems, then combines drums + bass + other to produce a
no-vocals (instrumental) track.
"""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf


def get_device_info() -> dict:
    """Return dict with 'device' ('cuda'/'cpu') and 'gpu_name' if applicable."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
            }
    except ImportError:
        pass
    return {"device": "cpu"}


def get_vocals_stem_path(song_path: str, output_dir: str | None = None) -> str | None:
    """Return the path to cached vocals.wav, or None if not available."""
    song_path = os.path.abspath(song_path)
    song_dir = os.path.dirname(song_path)
    song_name = os.path.splitext(os.path.basename(song_path))[0]

    if output_dir is None:
        output_dir = os.path.join(song_dir, f"{song_name}_stems")

    vocals_path = os.path.join(output_dir, "vocals.wav")
    if os.path.isfile(vocals_path):
        return vocals_path
    return None


def separate_song(
    audio_path: str,
    output_dir: str | None = None,
    device: str | None = None,
    callback: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Separate a song and return the instrumental (no-vocals) track.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory for cached stems. Defaults to
            ``<song_dir>/<song_name>_stems/``.
        device: 'cuda' or 'cpu'. Auto-detected if None.
        callback: Optional ``(message, fraction)`` progress callback.

    Returns:
        Tuple of (audio_data, sample_rate). audio_data is float32 numpy
        array with shape ``(samples, channels)``.
    """
    audio_path = os.path.abspath(audio_path)
    song_dir = os.path.dirname(audio_path)
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    if output_dir is None:
        output_dir = os.path.join(song_dir, f"{song_name}_stems")

    cached_path = os.path.join(output_dir, "no_vocals.wav")
    vocals_cached_path = os.path.join(output_dir, "vocals.wav")

    # Cache hit — only if both stems exist
    if os.path.isfile(cached_path) and os.path.isfile(vocals_cached_path):
        if callback:
            callback("Loading cached stems...", 0.0)
        data, sr = sf.read(cached_path, dtype="float32")
        if callback:
            callback("Loaded cached stems", 1.0)
        return data, sr

    # Auto-detect device
    if device is None:
        device = get_device_info()["device"]

    if callback:
        callback(f"Loading model ({device.upper()})...", 0.0)

    data, sr = _run_separation(audio_path, output_dir, cached_path, device, callback)

    return data, sr


def _run_separation(
    audio_path: str,
    output_dir: str,
    cached_path: str,
    device: str,
    callback: Callable[[str, float], None] | None,
) -> tuple[np.ndarray, int]:
    """Run Demucs separation, with GPU→CPU fallback on OOM."""
    import torch

    try:
        return _do_separate(audio_path, output_dir, cached_path, device, callback)
    except RuntimeError as exc:
        # CUDA OOM — fall back to CPU
        if device == "cuda" and "out of memory" in str(exc).lower():
            if callback:
                callback("GPU out of memory, falling back to CPU...", 0.0)
            torch.cuda.empty_cache()
            return _do_separate(audio_path, output_dir, cached_path, "cpu", callback)
        raise


def _do_separate(
    audio_path: str,
    output_dir: str,
    cached_path: str,
    device: str,
    callback: Callable[[str, float], None] | None,
) -> tuple[np.ndarray, int]:
    """Actually run the Demucs model and extract instrumental."""
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    model = get_model("htdemucs_ft")
    model.to(device)
    sr = model.samplerate

    if callback:
        callback(f"Separating ({device.upper()})...", 0.1)

    # Load audio as tensor (channels, samples) at model sample rate
    audio = AudioFile(Path(audio_path)).read(streams=0, samplerate=sr, channels=2)
    mix = audio.to(device)

    # Run separation — returns tensor (sources, channels, samples)
    sources = apply_model(model, mix[None], device=device, progress=False)[0]

    if callback:
        callback("Combining stems...", 0.8)

    # Combine drums + bass + other → instrumental (no vocals)
    # model.sources gives the stem order, e.g. ['drums', 'bass', 'other', 'vocals']
    instrumental = None
    for i, name in enumerate(model.sources):
        if name in ("drums", "bass", "other"):
            if instrumental is None:
                instrumental = sources[i].clone()
            else:
                instrumental += sources[i]

    if instrumental is None:
        raise RuntimeError("Demucs did not produce expected stems")

    # Convert tensor (channels, samples) → numpy (samples, channels) float32
    audio_data = instrumental.cpu().numpy().T.astype(np.float32)

    # Cache to disk
    if callback:
        callback("Caching stems...", 0.9)
    os.makedirs(output_dir, exist_ok=True)
    sf.write(cached_path, audio_data, sr)

    # Also cache the vocals stem for alignment
    for i, name in enumerate(model.sources):
        if name == "vocals":
            vocals_data = sources[i].cpu().numpy().T.astype(np.float32)
            sf.write(os.path.join(output_dir, "vocals.wav"), vocals_data, sr)
            break

    if callback:
        callback("Separation complete", 1.0)

    return audio_data, sr
