# Architecture & Technical Decisions — VocalForge

Stable design decisions for VocalForge's core systems. These are implemented and
unlikely to change. For active development context, see CLAUDE.md (project root).

---

## Alignment Algorithm

Use **cross-correlation** (`numpy.correlate` or `scipy.signal.correlate` with `mode='full'`):
1. Convert both tracks to mono if stereo
2. Use the minus track as reference, vocal recording as target
3. Find the lag at max correlation
4. Shift the vocal recording by that lag (zero-pad or trim)

For songs > 5 min, use FFT-based correlation (`scipy.signal.fftconvolve`) — O(n log n) vs O(n²).

### Constrained Alignment

Repetitive songs produce many correlation peaks of similar height. To avoid false matches:
1. Constrain the search window to ±max_offset (default ±30s)
2. Sanity check: warn if offset would clip >10% of either track
3. Multi-peak: prefer the peak closest to zero lag among near-equal candidates

### Chain Alignment

When a karaoke version differs slightly from the original:
1. Align minus-import to minus-sep (corrects karaoke-vs-original drift)
2. Align vocal-rec to vocal-sep (precise vocal timing)
3. minus-sep and vocal-sep are already aligned by definition (from same separation)

---

## LUFS Normalization

Target: **-14 LUFS** (Spotify/YouTube standard) as default, user-adjustable.

Workflow:
1. Measure integrated loudness of minus track and vocal track separately
2. Normalize both to target LUFS independently
3. Apply mixing ratio (e.g., 0.7 vocal + 1.0 instrumental)
4. Measure final mix loudness, apply final normalization to target

---

## Demucs Integration

- Use `htdemucs_ft` model (fine-tuned, best quality)
- Run in a QThread — separation of a 4-min song takes ~30-60s on GPU, ~3-5 min on CPU
- Show progress bar during separation
- Cache separated stems alongside the original file (e.g., `song_stems/` directory)
- Extract only the "no_vocals" (accompaniment) stem — discard drums/bass/other individual stems unless user wants them later

---

## Recording Workflow State Machine

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

---

## Track Model (5 Tracks + Mix Result)

| Track | Source | Alignment |
|-------|--------|-----------|
| **Original (plus)** | User imports the full song | Reference — not aligned, used for separation only |
| **Minus-sep** | Demucs separation of original | Anchor — all alignments reference this implicitly |
| **Vocal-sep** | Demucs separation of original | Aligned to minus-sep by definition |
| **Minus-import** | User imports karaoke version (optional) | Aligned to minus-sep via cross-correlation |
| **Vocal (recorded/imported)** | In-app recording or user import | Aligned to vocal-sep via cross-correlation |
| **Mix Result** | Auto-populated after export | No alignment — has alignment baked in |

---

## Vocal Enhancement Pipeline

See `VOCAL_ENHANCEMENT.md` (project root) for the full 13-stage processing chain
reference with algorithms, parameter tables, and tuning guidance.

Chain order: Gain Rider → De-Plosive → Noise Gate → Spectral NR → De-Reverb →
HPF → Parametric EQ → Compressor (Peak) → Compressor (Body) → De-Esser →
Soft Clipper → Reverb → Limiter.

All implementations use scipy/numpy only (MIT/BSD). `pedalboard` (GPL v3) is
incompatible with VocalForge's MIT license.
