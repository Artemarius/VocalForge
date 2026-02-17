"""Audio engine — device enumeration, playback, and recording.

Phase 1: device enumeration.
Phase 3: playback engine.
Phase 4: simultaneous playback + recording.
"""

import enum

import numpy as np
import sounddevice as sd


def _default_hostapi_devices() -> set[int]:
    """Return device indices belonging to the default host API."""
    try:
        api = sd.query_hostapis(sd.default.hostapi)
        return set(api["devices"])
    except Exception:
        return set()


def get_input_devices():
    """Return a list of available audio input devices.

    Each entry is a dict with keys: 'index', 'name', 'channels', 'sample_rate'.
    Only devices from the default host API are returned.
    """
    devices = sd.query_devices()
    api_devices = _default_hostapi_devices()
    result = []
    for i, dev in enumerate(devices):
        if i in api_devices and dev["max_input_channels"] > 0:
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
    Only devices from the default host API are returned.
    """
    devices = sd.query_devices()
    api_devices = _default_hostapi_devices()
    result = []
    for i, dev in enumerate(devices):
        if i in api_devices and dev["max_output_channels"] > 0:
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


class EngineState(enum.Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    RECORDING = "recording"


# Max recording duration: 15 minutes @ 44100 Hz mono float32
_MAX_REC_SECONDS = 15 * 60


class AudioEngine:
    """Manages audio playback and recording through sounddevice streams.

    Thread safety: _position, _volume, _rec_position are single Python values —
    reads/writes are atomic under the GIL. Callbacks only read/write shared
    state via atomic assignments. No locks needed.
    """

    def __init__(self):
        self._audio_data: np.ndarray | None = None
        self._sample_rate: int = 44100
        self._position: int = 0
        self._volume: float = 1.0
        self._state: EngineState = EngineState.IDLE
        self._stream: sd.OutputStream | None = None
        self._device_index: int | None = None
        self._channels: int = 2
        self._source_channels: int = 0
        self._total_frames: int = 0
        self._playback_ended: bool = False

        # Multi-track playback state
        self._multi_tracks: list[np.ndarray] = []
        self._multi_offsets: list[int] = []
        self._multi_gains: list[float] = []
        self._multi_mutes: list[bool] = []
        self._multi_mode: bool = False

        # Recording state
        self._input_stream: sd.InputStream | None = None
        self._input_device_index: int | None = None
        self._input_channels: int = 1
        self._rec_buffer: np.ndarray | None = None
        self._rec_position: int = 0
        self._rec_max_frames: int = 0
        self._latency_offset_ms: float = 0.0

    def load(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Load audio data for playback. Stops any active stream. Clears multi mode."""
        self.stop()
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._source_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        self._total_frames = audio_data.shape[0]
        self._position = 0
        self._multi_mode = False

    def load_multi(
        self,
        tracks: list[np.ndarray],
        sample_rate: int,
        offsets: list[int] | None = None,
        gains: list[float] | None = None,
        mutes: list[bool] | None = None,
    ) -> None:
        """Load multiple tracks for simultaneous playback.

        Args:
            tracks: List of audio arrays (each mono or stereo float32).
            sample_rate: Common sample rate for all tracks.
            offsets: Per-track start offset in samples (default all 0).
            gains: Per-track linear gain (default all 1.0).
            mutes: Per-track mute flags (default all False = audible).
        """
        self.stop()
        n = len(tracks)
        self._multi_tracks = list(tracks)
        self._multi_offsets = list(offsets) if offsets else [0] * n
        self._multi_gains = list(gains) if gains else [1.0] * n
        self._multi_mutes = list(mutes) if mutes else [False] * n
        self._sample_rate = sample_rate
        self._multi_mode = True
        self._position = 0

        # Total frames = max(track_length + offset)
        total = 0
        for i, track in enumerate(tracks):
            end = track.shape[0] + self._multi_offsets[i]
            if end > total:
                total = end
        self._total_frames = total
        # Set source channels to max across tracks for output channel count
        max_ch = 1
        for track in tracks:
            ch = 1 if track.ndim == 1 else track.shape[1]
            if ch > max_ch:
                max_ch = ch
        self._source_channels = max_ch

    def set_multi_mute(self, index: int, muted: bool) -> None:
        """Set the mute state for a multi-track slot. Safe during playback."""
        if 0 <= index < len(self._multi_mutes):
            self._multi_mutes[index] = muted

    def set_multi_gain(self, index: int, gain: float) -> None:
        """Set the gain for a multi-track slot. Safe during playback."""
        if 0 <= index < len(self._multi_gains):
            self._multi_gains[index] = gain

    def set_device(self, device_index: int, channels: int) -> None:
        """Set the output device. Channels are capped at 2."""
        self._device_index = device_index
        self._channels = min(channels, 2)

    def set_input_device(self, device_index: int, channels: int = 1) -> None:
        """Set the input device for recording. Only channel 0 is recorded."""
        self._input_device_index = device_index
        self._input_channels = max(1, channels)

    def play(self) -> None:
        """Start or resume playback."""
        if not self._multi_mode and self._audio_data is None:
            return
        if self._multi_mode and not self._multi_tracks:
            return

        if self._state == EngineState.PAUSED and self._stream is not None:
            self._state = EngineState.PLAYING
            self._playback_ended = False
            self._stream.start()
            return

        # Select callback based on mode
        callback = (
            self._multi_playback_callback
            if self._multi_mode
            else self._playback_callback
        )

        # Start fresh stream — preserve position for seamless restart
        saved_pos = self._position
        self.stop()
        self._position = saved_pos
        self._state = EngineState.PLAYING
        self._playback_ended = False
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            blocksize=1024,
            device=self._device_index,
            channels=self._channels,
            dtype="float32",
            callback=callback,
            finished_callback=self._on_playback_finished,
        )
        self._stream.start()

    def pause(self) -> None:
        """Pause playback, preserving position."""
        if self._stream is not None and self._state == EngineState.PLAYING:
            self._stream.stop()
            self._state = EngineState.PAUSED

    def stop(self) -> None:
        """Stop playback and reset position to 0."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._state = EngineState.IDLE
        self._position = 0
        self._playback_ended = False

    def seek(self, frame: int) -> None:
        """Set the playback position to a specific frame."""
        if self._audio_data is not None or self._multi_mode:
            self._position = max(0, min(frame, self._total_frames))

    def set_multi_offset(self, index: int, offset: int) -> None:
        """Update offset for a multi-track slot and recompute total_frames.

        Safe during playback — the callback reads offsets atomically.
        """
        if 0 <= index < len(self._multi_offsets):
            self._multi_offsets[index] = offset
            # Recompute total frames
            total = 0
            for i, track in enumerate(self._multi_tracks):
                end = track.shape[0] + self._multi_offsets[i]
                if end > total:
                    total = end
            self._total_frames = total

    # --- Recording ---

    def start_recording(self) -> None:
        """Start simultaneous playback + recording.

        Requires: loaded audio, IDLE state, input device set.
        Allocates a ring buffer and opens both output and input streams.
        """
        if self._audio_data is None:
            raise RuntimeError("No audio loaded for playback during recording")
        if self._state != EngineState.IDLE:
            raise RuntimeError(f"Cannot start recording in state {self._state.value}")
        if self._input_device_index is None:
            raise RuntimeError("No input device selected")

        # Allocate recording buffer (mono float32)
        self._rec_max_frames = self._sample_rate * _MAX_REC_SECONDS
        self._rec_buffer = np.zeros(self._rec_max_frames, dtype=np.float32)
        self._rec_position = 0
        self._position = 0
        self._playback_ended = False

        # Open output stream
        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                blocksize=1024,
                device=self._device_index,
                channels=self._channels,
                dtype="float32",
                callback=self._playback_callback,
                finished_callback=self._on_playback_finished,
            )
        except Exception:
            self._rec_buffer = None
            raise

        # Open input stream
        try:
            self._input_stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=1024,
                device=self._input_device_index,
                channels=self._input_channels,
                dtype="float32",
                callback=self._recording_callback,
            )
        except Exception:
            self._stream.close()
            self._stream = None
            self._rec_buffer = None
            raise

        # Start both streams
        self._state = EngineState.RECORDING
        self._stream.start()
        self._input_stream.start()

    def finish_recording(self) -> tuple[np.ndarray, int] | None:
        """Stop recording and return the captured audio.

        Applies latency offset: positive offset trims from the start,
        negative offset pads with silence at the start.

        Returns:
            (data, sample_rate) tuple, or None if no data was captured.
        """
        if self._state != EngineState.RECORDING:
            return None

        # Stop streams
        self._stop_streams()

        # Extract recorded data
        if self._rec_buffer is None or self._rec_position == 0:
            self._state = EngineState.IDLE
            self._rec_buffer = None
            return None

        raw = self._rec_buffer[:self._rec_position].copy()
        sr = self._sample_rate

        # Apply latency compensation
        offset_samples = int(self._latency_offset_ms * sr / 1000.0)
        if offset_samples > 0:
            # Positive offset: trim from the start
            if offset_samples < len(raw):
                raw = raw[offset_samples:]
            else:
                raw = np.array([], dtype=np.float32)
        elif offset_samples < 0:
            # Negative offset: pad silence at the start
            pad = np.zeros(-offset_samples, dtype=np.float32)
            raw = np.concatenate([pad, raw])

        # Clean up
        self._rec_buffer = None
        self._rec_position = 0
        self._state = EngineState.IDLE

        if len(raw) == 0:
            return None
        return raw, sr

    def stop_recording(self) -> None:
        """Stop recording and discard the buffer."""
        if self._state != EngineState.RECORDING:
            return

        self._stop_streams()
        self._rec_buffer = None
        self._rec_position = 0
        self._state = EngineState.IDLE

    def _stop_streams(self) -> None:
        """Stop and close both output and input streams."""
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._playback_ended = False

    # --- Properties ---

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
    def state(self) -> EngineState:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state == EngineState.PLAYING

    @property
    def is_paused(self) -> bool:
        return self._state == EngineState.PAUSED

    @property
    def is_recording(self) -> bool:
        return self._state == EngineState.RECORDING

    @property
    def playback_ended(self) -> bool:
        return self._playback_ended

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))

    @property
    def latency_offset_ms(self) -> float:
        return self._latency_offset_ms

    @latency_offset_ms.setter
    def latency_offset_ms(self, value: float) -> None:
        self._latency_offset_ms = max(-500.0, min(500.0, value))

    @property
    def rec_position(self) -> int:
        return self._rec_position

    # --- Callbacks ---

    def _playback_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio output callback — fills output buffer from loaded audio.

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

    def _recording_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio input callback — writes channel 0 to the ring buffer.

        Must be lock-free: only writes to _rec_buffer and _rec_position.
        """
        buf = self._rec_buffer
        pos = self._rec_position

        if buf is None:
            return

        remaining = self._rec_max_frames - pos
        if remaining <= 0:
            raise sd.CallbackStop

        valid = min(frames, remaining)
        buf[pos:pos + valid] = indata[:valid, 0]
        self._rec_position = pos + valid

    def _multi_playback_callback(
        self, outdata: np.ndarray, frames: int, time_info, status,
    ) -> None:
        """PortAudio callback for multi-track playback — sums unmuted tracks.

        Lock-free: reads pre-stored numpy arrays, does small multiplies,
        sums into outdata. No Python allocations beyond small numpy temps.
        """
        pos = self._position
        remaining = self._total_frames - pos

        if remaining <= 0:
            outdata[:] = 0
            raise sd.CallbackStop

        valid = min(frames, remaining)
        out_ch = outdata.shape[1]

        outdata[:] = 0

        tracks = self._multi_tracks
        offsets = self._multi_offsets
        gains = self._multi_gains
        mutes = self._multi_mutes

        for i in range(len(tracks)):
            if mutes[i]:
                continue

            track = tracks[i]
            offset = offsets[i]
            gain = gains[i]

            # Compute overlap of this track with the current output window
            track_start = pos - offset
            track_end = track_start + valid
            track_len = track.shape[0]

            if track_start >= track_len or track_end <= 0:
                continue

            # Clamp to valid track range
            src_start = max(0, track_start)
            src_end = min(track_len, track_end)
            dst_start = src_start - track_start
            dst_end = dst_start + (src_end - src_start)

            src_ch = 1 if track.ndim == 1 else track.shape[1]

            if track.ndim == 1:
                chunk = track[src_start:src_end] * gain
                for ch in range(out_ch):
                    outdata[dst_start:dst_end, ch] += chunk
            else:
                mix_ch = min(src_ch, out_ch)
                outdata[dst_start:dst_end, :mix_ch] += (
                    track[src_start:src_end, :mix_ch] * gain
                )

        if valid < frames:
            # Already zeroed from outdata[:] = 0 above
            self._position = pos + valid
            raise sd.CallbackStop

        self._position = pos + valid

    def _on_playback_finished(self) -> None:
        """Called by PortAudio when the output stream finishes (end of audio)."""
        self._playback_ended = True
        if self._state == EngineState.PLAYING:
            self._state = EngineState.IDLE


# Backward-compatible alias
PlaybackEngine = AudioEngine
