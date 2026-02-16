# CLAUDE.md — VocalForge

## What This Project Is

A Python desktop application for recording vocals over music tracks with automatic alignment, normalization, and mixing. Built with PySide6, sounddevice, Demucs, and pyloudnorm. Runs on Windows 10 (primary dev environment) and Linux.

**Read PROJECT.md for full roadmap and phase details.**

## Development Environment

- **OS:** Windows 10 Pro (primary), Linux (secondary)
- **Python:** 3.10+
- **IDE:** VS Code or command line
- **Audio hardware:** Behringer X-Air 16 mixer, Shure SM58 mic (not needed for development — any mic works)
- **GPU:** RTX 3060 6GB — available for Demucs inference but not required

## Project Structure

```
VocalForge/
├── vocalforge/
│   ├── __init__.py
│   ├── __main__.py          # Entry point: python -m vocalforge
│   ├── app.py               # QApplication setup, main window
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py   # Main window layout, panel orchestration
│   │   ├── import_panel.py  # Song loading, separation trigger
│   │   ├── record_panel.py  # Recording controls, device selection
│   │   ├── mix_panel.py     # Mixing controls, export
│   │   └── waveform.py      # Waveform display widget (shared)
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── engine.py        # Playback + recording streams (sounddevice)
│   │   ├── alignment.py     # Cross-correlation alignment
│   │   └── mixer.py         # LUFS normalization + mixing
│   ├── separation/
│   │   ├── __init__.py
│   │   └── demucs_worker.py # Demucs separation (runs in QThread)
│   └── utils/
│       ├── __init__.py
│       └── audio_io.py      # Load/save audio files (soundfile wrapper)
├── tests/
│   ├── test_alignment.py
│   ├── test_mixer.py
│   └── test_audio_io.py
├── requirements.txt
├── README.md
├── CLAUDE.md
├── LICENSE
└── .gitignore
```

## Architecture Rules

### Threading Model

Audio is timing-critical. Follow these rules strictly:

1. **Main thread** — PySide6 GUI only. Never do audio I/O or processing here.
2. **Audio callback thread** — sounddevice's PortAudio callback. Must be lock-free, no allocations, no Python GIL-heavy work. Only copy samples to/from pre-allocated numpy buffers.
3. **Processing thread(s)** — QThread for Demucs separation, alignment, mixing. Communicate with GUI via Qt signals/slots.

**Never call `sounddevice` functions from inside the audio callback.** The callback only reads/writes to shared numpy ring buffers.

### Module Boundaries

- `ui/` imports from `audio/` and `separation/` — never the reverse
- `audio/` and `separation/` must not import PySide6 (they can be tested headlessly)
- `separation/` must not import `audio/` — they are independent
- `utils/` is leaf — imports only standard library and soundfile

### Audio Format Conventions

- Internal format: **numpy float32 arrays**, shape `(samples, channels)` for stereo, `(samples,)` for mono
- Sample rate: **44100 Hz** as default, but respect source file sample rate — resample only when mixing tracks with different rates
- File I/O: use `soundfile.read()` / `soundfile.write()` — always specify `dtype='float32'`

### PySide6 Conventions

- Use PySide6, **not PyQt6** (LGPL vs GPL licensing)
- Import style: `from PySide6.QtWidgets import ...` (never `from PySide6 import *`)
- Connect signals with new-style syntax: `button.clicked.connect(self._on_click)`
- Long operations → QThread + signals, never `QApplication.processEvents()` hacks
- No QML — pure widgets (QMainWindow, QWidget, QVBoxLayout, etc.)

## Key Technical Decisions

### Alignment Algorithm

Use **cross-correlation** (`numpy.correlate` or `scipy.signal.correlate` with `mode='full'`):
1. Convert both tracks to mono if stereo
2. Use the minus track as reference, vocal recording as target
3. Find the lag at max correlation
4. Shift the vocal recording by that lag (zero-pad or trim)

For songs > 5 min, use FFT-based correlation (`scipy.signal.fftconvolve`) — O(n log n) vs O(n²).

### LUFS Normalization

Target: **-14 LUFS** (Spotify/YouTube standard) as default, user-adjustable.

Workflow:
1. Measure integrated loudness of minus track and vocal track separately
2. Normalize both to target LUFS independently
3. Apply mixing ratio (e.g., 0.7 vocal + 1.0 instrumental)
4. Measure final mix loudness, apply final normalization to target

### Demucs Integration

- Use `htdemucs_ft` model (fine-tuned, best quality)
- Run in a QThread — separation of a 4-min song takes ~30-60s on GPU, ~3-5 min on CPU
- Show progress bar during separation
- Cache separated stems alongside the original file (e.g., `song_stems/` directory)
- Extract only the "no_vocals" (accompaniment) stem — discard drums/bass/other individual stems unless user wants them later

### Recording Workflow State Machine

```
IDLE → [Start] → RECORDING → [Finish] → SAVE → PROCESSING → DONE
                      ↓
                   [Stop] → IDLE (discard recording)

Alternative (loaded vocal):
IDLE → [Load Vocal] → PROCESSING → DONE
```

- **IDLE:** minus track loaded, mic selected, ready
- **RECORDING:** minus plays through output device, mic records to buffer
- **SAVE:** raw recording auto-saved to disk as WAV (before any processing)
- **PROCESSING:** alignment + normalization + mixing (automatic). Vocal source can be a live recording or a loaded file — pipeline is identical.
- **DONE:** demo track ready, export button enabled

## Phase Boundaries

Development follows PROJECT.md phases. Each phase is self-contained:

- **Phase 1:** Project skeleton, PySide6 window, audio device enumeration
- **Phase 2:** Audio file loading + waveform display
- **Phase 3:** Playback engine (play/pause/stop minus track)
- **Phase 4:** Recording engine (simultaneous playback + recording)
- **Phase 5:** Alignment + mixing + export pipeline
- **Phase 6:** Demucs integration (source separation)
- **Phase 7:** UI polish, error handling, settings persistence
- **Phase 8:** Testing, packaging, README screenshots

Do not pull in components from later phases.

## Testing Strategy

- **Unit tests:** alignment math (known-offset synthetic signals), mixer output levels, audio I/O round-trip
- **Integration tests:** record silence → align → mix → verify output duration matches input
- **Manual validation:** record actual vocals, listen to output, verify alignment sounds correct
- **No mocking of sounddevice** in unit tests — audio/ module tests use synthetic numpy arrays only

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python -m vocalforge

# Run tests
pytest tests/ -v
```

## File Naming Conventions

- Python modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Test files: `test_<module>.py`
