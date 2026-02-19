# Completed Phases — VocalForge

Detailed descriptions of all completed development phases. For the active roadmap,
see PROJECT.md (project root).

---

## Phase 1 — Project Skeleton & Audio Devices [DONE]

**Goal:** PySide6 window opens, lists available audio input/output devices.

Tasks:
1. Create project structure per CLAUDE.md layout
2. `__main__.py` entry point — `python -m vocalforge`
3. `app.py` — QApplication, QMainWindow with placeholder panels
4. `audio/engine.py` — enumerate audio devices via `sounddevice.query_devices()`
5. `ui/record_panel.py` — dropdown for input device, dropdown for output device
6. `requirements.txt` with pinned versions

---

## Phase 2 — Audio File Loading & Waveform Display [DONE]

**Goal:** Load audio files and display waveforms.

Tasks:
1. `utils/audio_io.py` — load any format via soundfile, return float32 numpy array + sample rate
2. `ui/import_panel.py` — "Load Song", "Load Minus", "Load Vocal" buttons with file dialogs
3. `ui/waveform.py` — custom QWidget that draws a waveform from numpy array
4. Import panel shows waveforms for all three slots: song (plus), minus, and vocal
5. Handle edge cases: mono/stereo, different sample rates, very long files

Notes: All three track slots independently loadable. Audacity .aup3 import rejected (undocumented format).

---

## Phase 3 — Playback Engine [DONE]

**Goal:** Play loaded audio through selected output device.

Tasks:
1. `audio/engine.py` — `play(audio, device, sample_rate)` using `sounddevice.OutputStream`
2. Play/Pause/Stop controls in `ui/record_panel.py`
3. Playback position indicator on waveform display (moving cursor)
4. Volume control slider for playback

---

## Phase 4 — Simultaneous Playback + Recording [DONE]

**Goal:** Play minus track while recording microphone input.

Tasks:
1. Simultaneous playback + recording using `sounddevice.Stream` (full-duplex)
2. Ring buffer for recording (pre-allocated numpy array)
3. Recording state machine: IDLE → RECORDING → (Finish/Stop)
4. Start/Finish/Stop buttons with visual feedback
5. Auto-save raw recording as WAV immediately on "Finish"

---

## Phase 5 — Alignment, Mixing & Export [DONE]

**Goal:** Automatically align vocal to minus track, normalize, mix, and export.

Tasks:
1. `audio/alignment.py` — FFT cross-correlation, find lag, shift vocal
2. `audio/mixer.py` — LUFS measurement + normalization + mix ratio + peak limiting
3. `ui/mix_panel.py` — balance slider, target LUFS, Mix & Export button
4. Export as WAV or FLAC

---

## Phase 6 — Demucs Source Separation [DONE]

**Goal:** Extract instrumental from original song using Demucs htdemucs_ft model.

Tasks:
1. `separation/demucs_worker.py` — QThread worker, progress signals, stem caching
2. UI: "Separate" button, progress bar, auto-populate minus track
3. GPU detection with CPU fallback

---

## Phase 6b — Vocal Noise Reduction & Mix Balance [DONE]

**Goal:** Reduce vocal background noise, fix default mix balance.

Tasks:
1. `audio/noise_reduction.py` — spectral gating via noisereduce library
2. Integration into mix pipeline (after alignment, before normalization)
3. NR toggle + aggressiveness control in UI
4. Adjusted default vocal slider to better starting point

---

## Phase 6c — Chain Alignment, 5-Track Layout & Stem-Guided NR [DONE]

**Goal:** Chain alignment through separated stems, 5-track UI, stem-guided noise profiling.

Tasks:
1. 5-track UI layout (Original, Minus-sep, Vocal-sep, Minus-import, Vocal)
2. Both Demucs stems visible in UI after separation
3. Chain alignment: minus-import → minus-sep, vocal-rec → vocal-sep
4. Stem-guided noise profiling (silent regions from vocal-sep → noise reference)
5. Updated mix pipeline using chain alignment

---

## Phase 6d — Constrained Alignment, HPF & Mix Result Playback [DONE]

**Goal:** Fix alignment on repetitive songs, add HPF, in-app mix preview.

Tasks:
1. Constrained correlation window (±max_offset, default ±30s)
2. Multi-peak heuristic (prefer smallest offset among near-equal peaks)
3. 80 Hz high-pass filter (Butterworth via sosfiltfilt)
4. Mix Result track slot (6th track, auto-populated after export)

---

## Phase 7 — Interactive Alignment & Multi-Track Preview [DONE]

**Goal:** Decouple alignment from mixing, manual offset adjustment, multi-track playback.

Tasks:
1. Global playback cursor across all waveforms
2. Manual offset sliders (minus and vocal, ±60s range)
3. "Auto-Align" button (separate from Mix & Export)
4. Multi-track playback with per-track mute checkboxes
5. Waveform offset visualization

---

## Phase 7b — Bug Fixes & UI Polish [DONE]

**Goal:** Fix bugs from Phase 7 testing.

Fixes:
1. Mix & Export silent output bug
2. Vocal offset sign inversion
3. LUFS normalization after vocal enhancement
4. Offset sliders replacing number boxes
5. Clear button for imported minus
6. Waveform/fader overlap spacing
7. Space bar play/pause shortcut

---

## Phase 7c — Layout Fixes, Cursor Sync & Mix Reliability [DONE]

**Goal:** Fix remaining Phase 7b issues.

Fixes:
1. Playback cursor sync with per-track offsets
2. Mix corruption on second attempt
3. Minimum height for track groups (scrollable)
4. Reset Offsets button
5. Clear minus import resets minus offset

---

## Phase 7d — Multi-Track Seek, Mono Waveforms, Per-Offset Reset [DONE]

**Goal:** Five fixes from Phase 7c testing.

Fixes:
1. Multi-track seek misalignment (load ALL tracks, not just audible)
2. Stereo → mono waveform display
3. Mix Result chirp+silence (removed silent exception swallowing)
4. Per-offset reset buttons (replacing single Reset Offsets button)
5. Tests: multi-track engine + waveform (154 tests passing)

---

## Phase 8a — Signal Cleanup: Noise Gate + De-Reverb (v0.2.0) [DONE]

**Goal:** First two cleanup effects + preset infrastructure.

New effects:
1. Noise Gate (RMS-envelope gating with attack/release/hold)
2. De-reverb (spectral flux-based subtraction)

Additional features: Preset system (Raw/Clean/Enhanced), processed vocal export,
effects chain bypass, auto-tune volume, jump-to-click sliders, per-track level meters.
162 tests passing.

---

## Phase 8b — Tone & Dynamics: Parametric EQ + Compressor (v0.3.0) [DONE]

**Goal:** Tonal shaping and dynamics control.

New effects:
1. Parametric EQ (cascaded biquad filters, zero-phase, 3 presets: Clean Up/Warm/Bright)
2. Compressor (RMS envelope, soft-knee, auto makeup gain, parallel compression)

179 tests passing. v0.3.1 follow-up: NR mode selection (stationary/adaptive/auto),
advanced NR dialog, tuned defaults (HPF 100 Hz, EQ bright, limiter -0.7 dB). 189 tests.

---

## Phase 8c — Refinement: De-Esser + Reverb (v0.4.0) [DONE]

**Goal:** Complete the 9-stage vocal processing pipeline.

New effects:
1. De-esser (frequency-selective sibilance compressor, split-band/wideband modes)
2. Reverb (Schroeder algorithmic + optional convolution IR)

**9/9 effects working** — no more stubs. 203 tests passing.
