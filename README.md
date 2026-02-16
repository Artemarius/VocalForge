# VocalForge

Desktop app for recording vocals over music tracks with automatic alignment, normalization, and mixing.

## Motivation

My wife likes to sing. The workflow I had was: find a karaoke video, rip the audio in Audacity, record her singing over the minus track, then manually align three tracks (plus, minus, and vocal recording) by nudging waveforms until they match — every single time. It works, but it's tedious. VocalForge replaces all of that with a single app: load a song, optionally extract the instrumental automatically, hit record, and get a mixed demo track out the other end.

## Features

- **Load a song** — import any audio file as the "plus" (original with vocals)
- **Automatic vocal separation** — extract instrumental (minus) track using [Demucs](https://github.com/facebookresearch/demucs) (Meta's source separation model), or load your own minus track manually
- **Record vocals** — play the minus track through speakers/headphones while recording microphone input, with simple Start / Stop / Finish controls
- **Automatic alignment** — cross-correlation-based alignment of the vocal recording to the music track
- **LUFS normalization & mixing** — normalize loudness (ITU-R BS.1770-4) and blend vocal + instrumental at configurable ratios
- **Export demo track** — save the final mix as a WAV/FLAC file ready to share

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    PySide6 GUI                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  Import   │  │ Record   │  │  Mix & Export     │  │
│  │  Panel    │  │ Panel    │  │  Panel            │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
├───────┼──────────────┼─────────────────┼─────────────┤
│       │     Audio Engine (sounddevice)  │             │
│       │  ┌─────────┐  ┌────────────┐   │             │
│       │  │Playback  │  │ Recording  │   │             │
│       │  │Stream    │  │ Stream     │   │             │
│       │  └─────────┘  └────────────┘   │             │
├───────┼────────────────────────────────┼─────────────┤
│  Processing Pipeline                                  │
│  ┌───────────┐ ┌───────────┐ ┌──────────────────┐   │
│  │  Demucs   │ │ Alignment │ │ LUFS Normalize   │   │
│  │ Separator │ │ (xcorr)   │ │ + Mix (pyloudnorm│)  │
│  └───────────┘ └───────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Library | License |
|---|---|---|
| GUI | PySide6 | LGPL v3 |
| Audio playback & recording | sounddevice | MIT |
| Audio file I/O | soundfile | BSD 3-Clause |
| Vocal separation | Demucs (htdemucs_ft) | MIT |
| ML runtime | PyTorch | BSD 3-Clause |
| Track alignment | NumPy (cross-correlation) | BSD |
| LUFS normalization | pyloudnorm | MIT |

## Requirements

- Python 3.10+
- Audio interface with microphone input (tested with Behringer X-Air 16 + SM58)
- ~2 GB disk for Demucs model weights (downloaded on first use)
- GPU optional but recommended for faster source separation

## Installation

```bash
git clone https://github.com/artem-2024/VocalForge.git
cd VocalForge
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m vocalforge
```

## Usage

1. **Import** — Click "Load Song" and select the original track (plus). Optionally click "Separate" to extract the instrumental, or "Load Minus" to use your own backing track.
2. **Record** — Select your microphone input, click "Start". The minus track plays; sing along. Click "Finish" when done (or "Stop" to discard and retry).
3. **Mix** — Adjust the vocal/instrumental balance and target LUFS. Click "Export" to save the final mix.

## License

MIT — see [LICENSE](LICENSE).

> **Note:** VocalForge uses pre-trained Demucs models from Meta Research. The Demucs repository is MIT-licensed; however, the licensing of pre-trained model weights was not explicitly clarified separately by Meta before the repository was archived.
>
> Users are responsible for ensuring they have the rights to process any audio content used with this application.
