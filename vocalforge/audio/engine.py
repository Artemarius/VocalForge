"""Audio engine — device enumeration, playback, and recording.

Phase 1: device enumeration only.
"""

import sounddevice as sd


def get_input_devices():
    """Return a list of available audio input devices.

    Each entry is a dict with keys: 'index', 'name', 'channels', 'sample_rate'.
    """
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return result


def get_output_devices():
    """Return a list of available audio output devices.

    Each entry is a dict with keys: 'index', 'name', 'channels', 'sample_rate'.
    """
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return result


def get_default_input_device():
    """Return the index of the default input device, or None."""
    try:
        return sd.default.device[0]
    except Exception:
        return None


def get_default_output_device():
    """Return the index of the default output device, or None."""
    try:
        return sd.default.device[1]
    except Exception:
        return None
