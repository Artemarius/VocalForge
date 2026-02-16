"""Audio engine — device enumeration, playback, and recording.

Phase 1: device enumeration.
Phase 3: playback engine.
"""

import numpy as np
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


class PlaybackEngine:
    """Manages audio playback through a sounddevice OutputStream.

    Thread safety: _position and _volume are single Python values — reads/writes
    are atomic under the GIL. The callback only reads shared state; the main
    thread only writes it. No locks needed.
    """

    def __init__(self):
        self._audio_data: np.ndarray | None = None
        self._sample_rate: int = 44100
        self._position: int = 0
        self._volume: float = 1.0
        self._playing: bool = False
        self._paused: bool = False
        self._stream: sd.OutputStream | None = None
        self._device_index: int | None = None
        self._channels: int = 2
        self._source_channels: int = 0
        self._total_frames: int = 0

    def load(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Load audio data for playback. Stops any active stream."""
        self.stop()
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._source_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        self._total_frames = audio_data.shape[0]
        self._position = 0

    def set_device(self, device_index: int, channels: int) -> None:
        """Set the output device. Channels are capped at 2."""
        self._device_index = device_index
        self._channels = min(channels, 2)

    def play(self) -> None:
        """Start or resume playback."""
        if self._audio_data is None:
            return

        if self._paused and self._stream is not None:
            self._paused = False
            self._playing = True
            self._stream.start()
            return

        # Start fresh stream
        self.stop()
        self._playing = True
        self._paused = False
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            blocksize=1024,
            device=self._device_index,
            channels=self._channels,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()

    def pause(self) -> None:
        """Pause playback, preserving position."""
        if self._stream is not None and self._playing:
            self._stream.stop()
            self._playing = False
            self._paused = True

    def stop(self) -> None:
        """Stop playback and reset position to 0."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._playing = False
        self._paused = False
        self._position = 0

    def seek(self, frame: int) -> None:
        """Set the playback position to a specific frame."""
        if self._audio_data is not None:
            self._position = max(0, min(frame, self._total_frames))

    @property
    def position(self) -> int:
        return self._position

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback — fills output buffer from loaded audio.

        Must be lock-free: only reads _audio_data, _position, _volume;
        only writes _position.
        """
        pos = self._position
        data = self._audio_data
        vol = self._volume
        remaining = self._total_frames - pos

        if remaining <= 0:
            outdata[:] = 0
            raise sd.CallbackStop

        if remaining < frames:
            valid = remaining
        else:
            valid = frames

        if self._source_channels == 1:
            # Mono source → duplicate to all output channels
            chunk = data[pos:pos + valid] * vol
            for ch in range(outdata.shape[1]):
                outdata[:valid, ch] = chunk
        else:
            # Stereo or multi-channel source
            src_ch = min(self._source_channels, outdata.shape[1])
            outdata[:valid, :src_ch] = data[pos:pos + valid, :src_ch] * vol
            # Zero any extra output channels
            if src_ch < outdata.shape[1]:
                outdata[:valid, src_ch:] = 0

        # Zero-pad tail if we ran out of audio
        if valid < frames:
            outdata[valid:] = 0
            self._position = pos + valid
            raise sd.CallbackStop

        self._position = pos + valid

    def _on_stream_finished(self) -> None:
        """Called by PortAudio when the stream finishes (end of audio)."""
        self._playing = False
