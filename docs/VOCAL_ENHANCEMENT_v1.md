# Vocal Enhancement Reference — VocalForge

## Overview

This document describes the vocal processing chain for enhancing raw vocal recordings in VocalForge. The techniques are listed in the exact order they should be applied — order matters because each step feeds the next and incorrect ordering produces artifacts.

The core problem: recording vocals at home with an SM58 + mixer in an untreated room produces a raw signal with room reverb ("sounds like a well"), background noise, uneven dynamics, and tonal imbalances. The processing chain below transforms this into a clean, professional-sounding vocal that sits well over an instrumental backing track.

All implementations use MIT/BSD-licensed libraries only (scipy, numpy, noisereduce, pyloudnorm). Do NOT use Spotify's `pedalboard` library — it is GPL v3 and incompatible with VocalForge's MIT license.

---

## Processing Chain Order

```
Raw Recording
    │
    ▼
┌─────────────────────┐
│  1. Noise Gate       │  Remove silence gaps noise
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. Noise Reduction  │  Remove stationary background noise
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  3. De-Reverb        │  Reduce room reverb (the "well" sound)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4. High-Pass Filter │  Cut rumble below vocal range
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  5. Parametric EQ    │  Shape tonal balance
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  6. Compressor       │  Even out dynamics
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  7. De-Esser         │  Tame sibilance (sss/shh sounds)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  8. Reverb (optional)│  Add controlled, pleasant space
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  9. Limiter          │  Prevent clipping in final output
└─────────┬───────────┘
          ▼
    Enhanced Vocal
```

### Why This Order

1. **Gate and noise reduction first** — clean the signal before any tonal shaping. EQ and compression amplify noise if you process a dirty signal.
2. **De-reverb before EQ** — EQ after de-reverb avoids boosting reverb tails in the presence range.
3. **High-pass before parametric EQ** — removes the mud before surgical shaping, so the EQ decisions are clearer.
4. **EQ before compression** — shape the tone first, then even out the dynamics of the shaped signal. Compressing first would make EQ behave unpredictably (the compressor would react to frequencies you later remove).
5. **Compressor before de-esser** — compression can increase sibilance (it brings up quiet sibilants to the same level as louder ones), so de-ess after.
6. **Reverb after everything** — add space to the fully processed, clean vocal. Never apply reverb to a noisy or reverberant signal.
7. **Limiter absolutely last** — safety net to prevent digital clipping in the final output.

---

## Module Structure

```python
# vocalforge/audio/effects.py
#
# Each effect is a standalone function:
#   input:  numpy float32 array (samples,) or (samples, channels), sample_rate
#   output: numpy float32 array, same shape
#
# The chain function applies them in order:
#   process_vocal(audio, sr, config) -> processed_audio
```

---

## 1. Noise Gate

### Purpose
Silences the signal during pauses between vocal phrases. Without gating, the mic picks up room tone, AC hum, breathing, and ambient noise during gaps. When mixed over a quiet instrumental passage, this noise becomes audible.

### Algorithm
1. Compute RMS amplitude in short overlapping windows (10-30 ms)
2. Smooth the envelope with a low-pass filter to avoid chattering
3. Compare envelope to threshold
4. Apply gain: 1.0 where above threshold, 0.0 (or reduced, e.g., -40 dB) where below
5. Smooth the gain transitions using attack/release ramps to avoid clicks

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `threshold_db` | -60 to -20 | -35 | Level below which the gate closes. Set just above room noise floor |
| `attack_ms` | 0.5 to 10 | 2 | How fast the gate opens when signal exceeds threshold |
| `release_ms` | 20 to 500 | 100 | How fast the gate closes when signal drops below threshold |
| `hold_ms` | 0 to 200 | 50 | Minimum time the gate stays open after triggering (prevents chatter) |
| `reduction_db` | -inf to 0 | -40 | How much to attenuate gated signal. -inf = full mute, -20 = gentle |

### Implementation Notes
```python
import numpy as np

def noise_gate(audio, sr, threshold_db=-35, attack_ms=2, release_ms=100,
               hold_ms=50, reduction_db=-40):
    """
    Apply noise gate to audio signal.

    Steps:
    1. Compute RMS envelope in short windows
    2. Apply hold time to prevent rapid open/close
    3. Smooth with attack/release envelope follower
    4. Apply gain reduction where gate is closed
    """
    # Window for RMS calculation
    window_size = int(sr * 0.02)  # 20ms windows
    hop = window_size // 4

    # RMS envelope
    # Use stride tricks or simple loop for overlapping RMS
    # Compare to threshold (converted from dB to linear)
    threshold_linear = 10 ** (threshold_db / 20)
    reduction_linear = 10 ** (reduction_db / 20)

    # Attack/release as exponential smoothing coefficients
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))

    # Build gain envelope sample-by-sample (or block-by-block for speed)
    # Apply smoothed gain to audio
    pass
```

### Tuning Tips
- Record a few seconds of silence before singing — measure the RMS of that section to set the threshold automatically
- If words are getting clipped at the start, increase attack_ms
- If reverb tails are being cut off, increase release_ms or hold_ms
- For a more natural sound, use reduction_db=-20 instead of full muting — partial gating sounds less jarring

---

## 2. Noise Reduction (Spectral Gating)

### Purpose
Removes stationary background noise (hum, hiss, fan, computer noise) that persists even during singing. Unlike the noise gate which only helps during pauses, spectral gating removes noise that coexists with the vocal signal.

### Algorithm
The `noisereduce` library (MIT license) implements this:
1. Compute STFT of the entire signal
2. Estimate noise spectrum from a noise-only section (or statistically from the full signal)
3. For each time-frequency bin, if the energy is close to the noise floor, attenuate it
4. Reconstruct via inverse STFT

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `stationary` | bool | True | True = assumes constant noise profile. False = adapts over time |
| `prop_decrease` | 0.0 to 1.0 | 0.7 | How aggressively to reduce noise. 1.0 = full removal, risks artifacts |
| `n_std_thresh_stationary` | 0.5 to 3.0 | 1.5 | Number of std devs above noise mean to set threshold |
| `noise_clip` | ndarray or None | None | Separate noise-only sample for better profile estimation |

### Implementation
```python
import noisereduce as nr

def reduce_noise(audio, sr, noise_clip=None, prop_decrease=0.7, stationary=True):
    """
    Spectral gating noise reduction.

    If noise_clip is provided (e.g., a few seconds of room silence recorded
    before singing), use it for noise profile estimation — gives much better
    results than statistical estimation from the full signal.
    """
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        prop_decrease=prop_decrease,
        stationary=stationary,
        n_std_thresh_stationary=1.5,
    )
```

### Tuning Tips
- **Best practice:** record 3-5 seconds of silence (room tone) before singing. Pass this as `noise_clip` for dramatically better noise estimation
- Start with `prop_decrease=0.6` and increase gradually. Above 0.85 you'll start hearing "musical noise" (tinkling, underwater artifacts)
- For non-stationary noise (e.g., passing traffic), set `stationary=False`
- The noisereduce PyTorch backend is faster for longer recordings

### VocalForge Integration
Consider adding a "Record Room Tone" button before the main recording starts — 3 seconds of silence capture. Store this as the noise reference. This single addition dramatically improves noise reduction quality.

---

## 3. De-Reverb

### Purpose
Reduces room reverb baked into the recording. This is the most impactful enhancement for the "well" problem. Room reverb is different from noise: it's not additive/stationary, but convolutive — each sound gets smeared by the room's impulse response, creating a wash that muddies clarity.

### Approach A: Spectral Subtraction (Simple, No ML)

This approach attenuates the reverb tail by analyzing the decay characteristics of the signal.

#### Algorithm
1. Compute STFT of the signal
2. Estimate the reverb-to-direct ratio in each frequency band by analyzing temporal decay
3. Apply a spectral mask that attenuates the reverberant energy while preserving direct sound
4. Reconstruct via inverse STFT

```python
import numpy as np
from scipy.signal import stft, istft

def dereverb_spectral(audio, sr, strength=0.5, frame_size=2048, hop_size=512):
    """
    Simple spectral de-reverb using transient-to-steady-state ratio.

    Principle: direct sound has sharp transients, reverb has smooth decay.
    We enhance the transient portions and attenuate the sustained/decaying portions.

    strength: 0.0 = no effect, 1.0 = maximum de-reverb (may sound thin)
    """
    f, t, Zxx = stft(audio, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Compute spectral flux (frame-to-frame change) as a transient indicator
    # High flux = direct sound (transient), low flux = reverb tail (sustained)
    flux = np.zeros_like(magnitude)
    flux[:, 1:] = np.maximum(0, magnitude[:, 1:] - magnitude[:, :-1])

    # Normalize flux per frequency band
    flux_normalized = np.zeros_like(flux)
    for i in range(flux.shape[0]):
        max_flux = np.max(flux[i])
        if max_flux > 0:
            flux_normalized[i] = flux[i] / max_flux

    # Create mask: preserve transients (high flux), attenuate sustained (low flux)
    mask = (1 - strength) + strength * flux_normalized
    mask = np.clip(mask, 0.1, 1.0)  # Never fully zero — preserves some natural decay

    # Optional: smooth mask temporally to avoid artifacts
    from scipy.ndimage import uniform_filter1d
    mask = uniform_filter1d(mask, size=3, axis=1)

    # Apply mask and reconstruct
    Zxx_processed = magnitude * mask * np.exp(1j * phase)
    _, audio_out = istft(Zxx_processed, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size)

    return audio_out[:len(audio)]
```

#### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `strength` | 0.0 to 1.0 | 0.4 | How aggressively to reduce reverb. Start low |
| `frame_size` | 1024 to 4096 | 2048 | STFT frame size. Larger = better frequency resolution, worse time resolution |

#### Limitations
This is a simplified approach. It works for moderate room reverb but won't perform miracles on heavily reverberant recordings. For those, the ML approach (below) is significantly better.

### Approach B: ML-Based (UVR/MDX-Net)

The Ultimate Vocal Remover project has a dedicated reverb removal model (MDX-Net "Reverb HQ") that is significantly more effective than spectral methods. UVR is MIT-licensed.

This is a heavier dependency (separate model download, ~200 MB) but produces dramatically better results. Consider this as an optional advanced enhancement that the user can enable.

#### Integration Pattern
Same pattern as Demucs — run in a QThread, download model on first use, cache results.

### Tuning Tips
- **Start with Approach A at strength 0.3-0.4** — for moderate room reverb this may be enough
- If the recording still sounds "roomy" after spectral de-reverb, increase strength or switch to Approach B
- **Never de-reverb to complete dryness** — a totally dry vocal sounds unnaturally close and clinical. Some residual room sound is fine, especially if you'll add controlled reverb in step 8
- Listen on headphones when tuning de-reverb — room monitors add their own reverb and mask the effect

---

## 4. High-Pass Filter (Low Cut)

### Purpose
Removes low-frequency content below the vocal range: rumble, foot stomps, AC vibration, handling noise, and the SM58's proximity effect (bass boost when singing close to the mic). This is almost universally applied to vocals and there's rarely a reason to skip it.

### Algorithm
Butterworth high-pass filter applied via second-order sections (SOS) for numerical stability.

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `cutoff_hz` | 60 to 200 | 100 | Cutoff frequency. 80-100 Hz for male, 120-150 Hz for female |
| `order` | 2 to 6 | 4 | Filter order. 4th order = 24 dB/octave rolloff, clean and standard |

### Implementation
```python
from scipy.signal import butter, sosfilt

def highpass_filter(audio, sr, cutoff_hz=100, order=4):
    """
    Butterworth high-pass filter.

    Use zero-phase filtering (sosfiltfilt) for offline processing to avoid
    phase distortion. For real-time, use sosfilt (causal, introduces phase shift).
    """
    from scipy.signal import sosfiltfilt
    sos = butter(order, cutoff_hz, btype='high', fs=sr, output='sos')
    return sosfiltfilt(sos, audio).astype(np.float32)
```

### Tuning Tips
- Use `sosfiltfilt` (zero-phase) instead of `sosfilt` since we're doing offline processing — this eliminates phase distortion and produces a perfectly symmetrical filter response
- For the SM58 with close mic technique, you may want to go as high as 120-150 Hz even for male vocals to counteract the proximity effect
- If the vocal sounds thin after HPF, the cutoff is too high — lower it by 20 Hz

---

## 5. Parametric EQ

### Purpose
Shapes the tonal balance of the voice. Every voice has different problem areas: room modes that create muddiness, nasal resonances, lack of clarity or brightness. EQ is where you fix these and make the voice sound polished.

### Common Vocal EQ Moves

| Frequency Range | Action | Reason |
|---|---|---|
| 200-400 Hz | Cut 2-4 dB, wide Q (0.7-1.0) | Reduces muddiness and "boxiness" — the main source of the "well" sound |
| 400-800 Hz | Cut 1-3 dB if nasal, narrow Q (2-4) | Reduces "honky" or nasal quality |
| 1-2 kHz | Leave flat or gentle cut | Avoid harshness region |
| 2.5-5 kHz | Boost 1-3 dB, wide Q (0.5-1.0) | Adds "presence" — voice cuts through the mix |
| 6-8 kHz | Cut if sibilant (handled by de-esser) | Only if de-esser isn't enough |
| 8-12 kHz | Boost 1-2 dB, shelf | Adds "air" — brightness and shimmer |

### Principle: Cut Narrow, Boost Wide
When removing problems (mud, harshness, resonance), use a narrow Q (high selectivity, surgical). When enhancing (presence, air), use a wide Q (gentle, natural-sounding). This is the universal rule of professional vocal EQ.

### Algorithm
Cascaded biquad filters. Each band is a second-order IIR filter with adjustable center frequency, gain, and Q (bandwidth).

### Parameters Per Band
| Parameter | Range | Description |
|---|---|---|
| `freq_hz` | 20 to 20000 | Center frequency of the band |
| `gain_db` | -12 to +12 | Boost (positive) or cut (negative) |
| `q` | 0.1 to 10 | Bandwidth. Low Q = wide, high Q = narrow |
| `filter_type` | peak/lowshelf/highshelf | Type of filter shape |

### Implementation
```python
from scipy.signal import iirpeak, iirnotch, sosfilt, sosfiltfilt
import numpy as np

def biquad_peak(freq_hz, gain_db, q, sr):
    """
    Design a peaking EQ biquad filter.
    Returns second-order section (SOS) coefficients.
    """
    A = 10 ** (gain_db / 40)  # amplitude
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    # Normalize
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    # Convert to SOS format for cascading
    return np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])

def biquad_shelf(freq_hz, gain_db, q, sr, shelf_type='high'):
    """
    Design a shelving EQ biquad filter.
    shelf_type: 'high' for high shelf, 'low' for low shelf.
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    two_sqrt_A_alpha = 2 * np.sqrt(A) * alpha

    if shelf_type == 'low':
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha
    else:  # high shelf
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def parametric_eq(audio, sr, bands):
    """
    Apply parametric EQ with multiple bands.

    bands: list of dicts, each with:
        - freq_hz: center frequency
        - gain_db: gain (negative = cut, positive = boost)
        - q: bandwidth (0.1 = very wide, 10 = very narrow)
        - type: 'peak', 'lowshelf', or 'highshelf'

    Example:
        bands = [
            {'freq_hz': 300,  'gain_db': -3,  'q': 0.8, 'type': 'peak'},      # cut mud
            {'freq_hz': 3500, 'gain_db': 2,   'q': 0.7, 'type': 'peak'},      # add presence
            {'freq_hz': 10000,'gain_db': 1.5, 'q': 0.7, 'type': 'highshelf'}, # add air
        ]
    """
    # Cascade all bands into a single SOS array
    all_sos = []
    for band in bands:
        if band.get('gain_db', 0) == 0:
            continue  # skip bands with no gain change
        if band['type'] == 'peak':
            sos = biquad_peak(band['freq_hz'], band['gain_db'], band['q'], sr)
        elif band['type'] in ('lowshelf', 'highshelf'):
            sos = biquad_shelf(band['freq_hz'], band['gain_db'], band['q'], sr,
                              shelf_type=band['type'].replace('shelf', ''))
        all_sos.append(sos)

    if not all_sos:
        return audio

    sos = np.vstack(all_sos)
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos, audio).astype(np.float32)
```

### Preset Suggestions

**"Clean Up" preset (conservative, safe for any voice):**
```python
clean_up_bands = [
    {'freq_hz': 250,   'gain_db': -2.5, 'q': 0.8, 'type': 'peak'},      # reduce mud
    {'freq_hz': 3500,  'gain_db': 1.5,  'q': 0.7, 'type': 'peak'},      # presence
    {'freq_hz': 10000, 'gain_db': 1.0,  'q': 0.7, 'type': 'highshelf'}, # air
]
```

**"Warm" preset (for thin-sounding recordings):**
```python
warm_bands = [
    {'freq_hz': 200,   'gain_db': 1.5,  'q': 0.6, 'type': 'lowshelf'},  # add warmth
    {'freq_hz': 800,   'gain_db': -2.0, 'q': 1.5, 'type': 'peak'},      # reduce honk
    {'freq_hz': 4000,  'gain_db': 1.0,  'q': 0.7, 'type': 'peak'},      # gentle presence
]
```

**"Bright" preset (for dull recordings or voices that need to cut through):**
```python
bright_bands = [
    {'freq_hz': 300,   'gain_db': -3.0, 'q': 0.8, 'type': 'peak'},      # cut mud
    {'freq_hz': 3000,  'gain_db': 2.5,  'q': 0.6, 'type': 'peak'},      # strong presence
    {'freq_hz': 8000,  'gain_db': 2.0,  'q': 0.7, 'type': 'highshelf'}, # bright air
]
```

---

## 6. Compressor

### Purpose
Reduces dynamic range — makes quiet parts louder and loud parts quieter. Without compression, some words disappear into the backing track while others jump out too loudly. Compression makes the vocal level consistent and "present" throughout the song.

### Algorithm
1. Compute the signal envelope (RMS in short windows)
2. Convert to dB
3. Apply gain reduction curve: above the threshold, reduce gain by the ratio
4. Smooth the gain reduction with attack/release envelope
5. Apply the computed gain to the original signal
6. Add makeup gain to restore overall level

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `threshold_db` | -40 to 0 | -18 | Level above which compression starts |
| `ratio` | 1.0 to 20.0 | 3.0 | Compression ratio. 3:1 = gentle, 8:1+ = aggressive |
| `attack_ms` | 0.1 to 100 | 15 | How fast the compressor reacts to loud signals |
| `release_ms` | 10 to 1000 | 200 | How fast the compressor stops compressing after signal drops |
| `makeup_db` | 0 to 20 | auto | Gain added after compression to restore volume |
| `knee_db` | 0 to 12 | 6 | Soft knee width. 0 = hard knee, 6+ = gradual transition |

### Implementation
```python
def compressor(audio, sr, threshold_db=-18, ratio=3.0, attack_ms=15,
               release_ms=200, makeup_db=None, knee_db=6):
    """
    Dynamic range compressor.

    Uses RMS envelope detection with exponential attack/release smoothing.
    Soft knee for more natural compression.
    """
    # Envelope detection
    window_samples = int(sr * 0.01)  # 10ms RMS window
    hop = window_samples // 2

    # Compute RMS envelope (block-based, then interpolate to sample rate)
    n_blocks = len(audio) // hop
    rms_db = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        end = min(start + window_samples, len(audio))
        block_rms = np.sqrt(np.mean(audio[start:end] ** 2) + 1e-10)
        rms_db[i] = 20 * np.log10(block_rms + 1e-10)

    # Gain computation with soft knee
    gain_db = np.zeros_like(rms_db)
    for i in range(len(rms_db)):
        level = rms_db[i]
        if knee_db > 0:
            # Soft knee
            knee_start = threshold_db - knee_db / 2
            knee_end = threshold_db + knee_db / 2
            if level < knee_start:
                gain_db[i] = 0  # no compression
            elif level > knee_end:
                gain_db[i] = (threshold_db - level) * (1 - 1/ratio)
            else:
                # Quadratic interpolation in knee region
                x = level - knee_start
                gain_db[i] = ((1/ratio - 1) * x * x) / (2 * knee_db)
        else:
            # Hard knee
            if level > threshold_db:
                gain_db[i] = (threshold_db - level) * (1 - 1/ratio)

    # Attack/release smoothing
    attack_coeff = np.exp(-1.0 / (sr / hop * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr / hop * release_ms / 1000))

    smoothed = np.zeros_like(gain_db)
    smoothed[0] = gain_db[0]
    for i in range(1, len(gain_db)):
        if gain_db[i] < smoothed[i-1]:  # gain decreasing = attacking
            smoothed[i] = attack_coeff * smoothed[i-1] + (1 - attack_coeff) * gain_db[i]
        else:  # gain increasing = releasing
            smoothed[i] = release_coeff * smoothed[i-1] + (1 - release_coeff) * gain_db[i]

    # Interpolate gain envelope to sample rate
    from scipy.interpolate import interp1d
    time_blocks = np.arange(len(smoothed)) * hop
    interp_func = interp1d(time_blocks, smoothed, kind='linear',
                           fill_value='extrapolate')
    gain_sample = interp_func(np.arange(len(audio)))

    # Apply gain
    gain_linear = 10 ** (gain_sample / 20)
    output = audio * gain_linear

    # Auto makeup gain: compensate for average gain reduction
    if makeup_db is None:
        avg_reduction = np.mean(smoothed)
        makeup_db = -avg_reduction * 0.7  # compensate ~70% of average reduction
    output *= 10 ** (makeup_db / 20)

    return output.astype(np.float32)
```

### Parallel Compression (Advanced)

For a more natural, "thick" vocal sound, mix the compressed signal with the original:

```python
def parallel_compress(audio, sr, mix=0.6, **compressor_kwargs):
    """
    Parallel (NY-style) compression.
    mix: 0.0 = all dry, 1.0 = all compressed. 0.5-0.7 is typical.
    """
    compressed = compressor(audio, sr, **compressor_kwargs)
    return ((1 - mix) * audio + mix * compressed).astype(np.float32)
```

This preserves natural dynamics (from the dry signal) while adding body and consistency (from the compressed signal). Use this instead of regular compression if the regular compressor sounds too "squashed."

### Tuning Tips
- **Attack time is critical for vocals.** Too fast (< 5 ms) kills the natural attack of consonants, making singing sound dull. Too slow (> 50 ms) lets transients through uncontrolled. 10-20 ms is the sweet spot for singing.
- **Ratio 3:1** is the standard starting point for vocal compression. Only go higher (6:1+) if you want a deliberately compressed, "radio" sound.
- **Listen for "pumping"** — a rhythmic volume swell that follows the compression. It means the release is too fast. Increase release_ms.
- **2-6 dB of gain reduction** is typical. If you're seeing 10+ dB, the threshold is too low.

---

## 7. De-Esser

### Purpose
Tames sibilance — the harsh "sss", "shh", "tch", "f" sounds that become piercing in recordings. The SM58 is not particularly bright, but EQ presence boosts (step 5) and compression (step 6) can both increase sibilance. The de-esser catches it after those steps.

### Algorithm
A de-esser is a frequency-selective compressor:
1. Split the signal into sibilant band (bandpass 4-10 kHz) and the rest
2. Detect the envelope of the sibilant band
3. When sibilant energy exceeds a threshold, apply gain reduction to the full signal (or just the sibilant band — "split-band" mode)
4. Smooth the gain reduction to avoid clicks

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `freq_hz` | 3000 to 10000 | 6000 | Center of the sibilant detection band |
| `bandwidth_hz` | 1000 to 6000 | 4000 | Width of the detection band |
| `threshold_db` | -40 to 0 | -20 | Level above which de-essing engages |
| `reduction_db` | 0 to 12 | 6 | Maximum dB of sibilance reduction |
| `mode` | 'wideband' / 'split' | 'split' | Wideband: reduces full signal. Split: reduces only sibilant band |

### Implementation
```python
from scipy.signal import butter, sosfiltfilt

def de_esser(audio, sr, freq_hz=6000, bandwidth_hz=4000, threshold_db=-20,
             reduction_db=6, mode='split'):
    """
    Frequency-selective compressor targeting sibilance.

    'split' mode (recommended): only reduces gain in the sibilant frequency band,
    preserving the body of the vocal. More transparent than wideband.
    """
    # Design bandpass for sibilant detection
    low = max(freq_hz - bandwidth_hz / 2, 100)
    high = min(freq_hz + bandwidth_hz / 2, sr / 2 - 100)
    sos_bp = butter(4, [low, high], btype='band', fs=sr, output='sos')

    # Extract sibilant band
    sibilant = sosfiltfilt(sos_bp, audio)

    # Envelope of sibilant band
    envelope = np.abs(sibilant)
    # Smooth envelope (~5ms window)
    smooth_samples = int(sr * 0.005)
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode='same')

    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-10)

    # Compute gain reduction
    gain_db = np.zeros_like(envelope_db)
    above_threshold = envelope_db > threshold_db
    gain_db[above_threshold] = np.clip(
        -(envelope_db[above_threshold] - threshold_db),
        -reduction_db, 0
    )

    # Smooth gain changes (fast attack, medium release)
    attack = np.exp(-1.0 / (sr * 0.001))   # 1ms attack
    release = np.exp(-1.0 / (sr * 0.050))  # 50ms release
    smoothed = np.zeros_like(gain_db)
    for i in range(1, len(gain_db)):
        coeff = attack if gain_db[i] < smoothed[i-1] else release
        smoothed[i] = coeff * smoothed[i-1] + (1 - coeff) * gain_db[i]

    gain_linear = 10 ** (smoothed / 20)

    if mode == 'split':
        # Apply gain only to sibilant band, add back the rest
        rest = audio - sibilant
        return (rest + sibilant * gain_linear).astype(np.float32)
    else:
        # Wideband: apply to entire signal
        return (audio * gain_linear).astype(np.float32)
```

### Tuning Tips
- **Female vocals:** sibilance typically lives around 5-8 kHz
- **Male vocals:** typically 4-7 kHz
- **Split-band mode** is almost always better — wideband de-essing can make the entire signal dull during sibilant moments
- **Over-de-essing** produces a lisp effect ("s" sounds become "th"). If this happens, reduce `reduction_db` or raise `threshold_db`
- Listen to isolated sibilants (the "sss" sounds) to set the frequency — if they sound dull after processing, the frequency is too low

---

## 8. Reverb (Controlled, Optional)

### Purpose
After removing the room's natural (bad) reverb in step 3, the vocal may sound unnaturally dry and close — like the singer is inside your head. Adding a small amount of controlled, pleasant reverb places the voice in a nice acoustic space while keeping it clear.

### Algorithm: Convolution Reverb
Convolve the signal with an impulse response (IR) recorded in a real acoustic space (concert hall, studio, plate reverb unit). This is the most realistic reverb method and trivially implemented.

```python
from scipy.signal import fftconvolve
import soundfile as sf

def convolution_reverb(audio, sr, ir_path, wet_mix=0.15, predelay_ms=25,
                       ir_highcut_hz=8000):
    """
    Convolution reverb using a real impulse response.

    ir_path: path to impulse response WAV file
    wet_mix: 0.0 = fully dry, 1.0 = fully wet. 0.10-0.25 typical for vocals
    predelay_ms: gap before reverb starts. Preserves vocal clarity
    ir_highcut_hz: roll off high frequencies in the IR to avoid sibilant reverb
    """
    # Load impulse response
    ir, ir_sr = sf.read(ir_path, dtype='float32')
    if ir.ndim > 1:
        ir = ir.mean(axis=1)  # mono

    # Resample IR if needed
    if ir_sr != sr:
        from scipy.signal import resample
        ir = resample(ir, int(len(ir) * sr / ir_sr))

    # High-cut the IR to prevent sibilant/harsh reverb tails
    if ir_highcut_hz:
        sos = butter(4, ir_highcut_hz, btype='low', fs=sr, output='sos')
        ir = sosfiltfilt(sos, ir).astype(np.float32)

    # Normalize IR
    ir = ir / (np.max(np.abs(ir)) + 1e-10)

    # Convolve
    wet = fftconvolve(audio, ir, mode='full')[:len(audio)]

    # Pre-delay: shift the wet signal forward in time
    predelay_samples = int(sr * predelay_ms / 1000)
    if predelay_samples > 0:
        wet = np.concatenate([np.zeros(predelay_samples), wet[:-predelay_samples]])

    # Mix
    output = (1 - wet_mix) * audio + wet_mix * wet
    return output.astype(np.float32)
```

### Free Impulse Response Sources
- **EchoThief** (http://www.echothief.com/) — 115 free IRs from real spaces (Creative Commons)
- **Open AIR** (https://openairlib.net/) — academic collection of impulse responses (various CC licenses)
- **Voxengo** (https://www.voxengo.com/free/impulseresponses/) — free IRs from real rooms

For vocals over pop/rock instrumentals, a short plate reverb or small room IR works best. Long cathedral-style IRs will wash the vocal out.

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `wet_mix` | 0.0 to 0.5 | 0.15 | Reverb amount. If you can clearly hear the reverb, it's too much |
| `predelay_ms` | 0 to 80 | 25 | Gap before reverb. Separates dry voice from reverb tail |
| `ir_highcut_hz` | 4000 to 16000 | 8000 | Roll off highs in reverb to prevent sibilant wash |

### Algorithmic Reverb Alternative

If you don't want to bundle IR files, a simple Schroeder reverb (allpass + comb filters) works:

```python
def schroeder_reverb(audio, sr, decay=0.7, wet_mix=0.15, predelay_ms=25):
    """
    Simple algorithmic reverb using Schroeder's design.
    4 parallel comb filters -> 2 series allpass filters.

    Lighter than convolution, less realistic but fully self-contained.
    """
    def comb_filter(signal, delay_samples, feedback):
        output = np.zeros(len(signal) + delay_samples)
        for i in range(len(signal)):
            output[i] += signal[i]
            output[i + delay_samples] += output[i] * feedback
        return output[:len(signal)]

    def allpass_filter(signal, delay_samples, feedback):
        output = np.zeros_like(signal)
        buffer = np.zeros(delay_samples)
        buf_idx = 0
        for i in range(len(signal)):
            delayed = buffer[buf_idx]
            output[i] = -feedback * signal[i] + delayed
            buffer[buf_idx] = signal[i] + feedback * delayed
            buf_idx = (buf_idx + 1) % delay_samples
        return output

    # Comb filter delays (in samples, at 44100 Hz, scaled for other rates)
    scale = sr / 44100
    comb_delays = [int(d * scale) for d in [1557, 1617, 1491, 1422]]
    allpass_delays = [int(d * scale) for d in [225, 556]]

    # Parallel comb filters
    combs = sum(comb_filter(audio, d, decay) for d in comb_delays) / len(comb_delays)

    # Series allpass filters
    wet = combs
    for d in allpass_delays:
        wet = allpass_filter(wet, d, 0.5)

    # Pre-delay
    predelay_samples = int(sr * predelay_ms / 1000)
    if predelay_samples > 0:
        wet = np.concatenate([np.zeros(predelay_samples), wet[:-predelay_samples]])

    return ((1 - wet_mix) * audio + wet_mix * wet).astype(np.float32)
```

### Tuning Tips
- **The golden rule: if you can hear the reverb, it's too much.** Reverb on vocals should be felt (adds space and polish) not heard (washes out the voice). For home karaoke demos, 10-20% wet is typical.
- **Pre-delay is underrated.** Even 20-30 ms of pre-delay dramatically improves clarity because the dry voice and the reverb don't overlap at the start of each word.
- **Always high-cut the reverb** (8 kHz or lower). Bright reverb tails add harshness and interfere with sibilance control.

---

## 9. Limiter

### Purpose
Safety net at the end of the chain. After all the processing — especially compression makeup gain, EQ boosts, and reverb — the signal may occasionally exceed 0 dBFS (digital full scale), causing clipping distortion. The limiter catches these peaks.

### Algorithm
A limiter is just a compressor with a very high ratio (∞:1) and very fast attack:
1. Detect peaks above the ceiling
2. Apply instantaneous gain reduction to prevent exceeding the ceiling
3. Very fast attack (< 1 ms), moderate release (50-100 ms)

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `ceiling_db` | -3 to 0 | -0.5 | Maximum output level. -0.5 dB is standard (leaves headroom for encoding) |
| `release_ms` | 10 to 200 | 50 | How fast the limiter releases after a peak |

### Implementation
```python
def limiter(audio, sr, ceiling_db=-0.5, release_ms=50):
    """
    Brick-wall peak limiter.
    Prevents output from exceeding ceiling_db.
    """
    ceiling_linear = 10 ** (ceiling_db / 20)

    # Lookahead: delay the signal slightly so we can anticipate peaks
    lookahead_samples = int(sr * 0.005)  # 5ms lookahead

    # Peak detection with lookahead
    peak_env = np.zeros(len(audio))
    for i in range(len(audio)):
        window_end = min(i + lookahead_samples, len(audio))
        peak_env[i] = np.max(np.abs(audio[i:window_end]))

    # Compute gain reduction
    gain = np.ones(len(audio))
    above_ceiling = peak_env > ceiling_linear
    gain[above_ceiling] = ceiling_linear / peak_env[above_ceiling]

    # Smooth release (attack is instant)
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    for i in range(1, len(gain)):
        if gain[i] > gain[i-1]:  # releasing
            gain[i] = release_coeff * gain[i-1] + (1 - release_coeff) * gain[i]

    # Apply with lookahead delay compensation
    output = np.zeros_like(audio)
    output[lookahead_samples:] = audio[:-lookahead_samples] * gain[lookahead_samples:]
    output[:lookahead_samples] = audio[:lookahead_samples] * gain[:lookahead_samples]

    return output.astype(np.float32)
```

### Tuning Tips
- **Set ceiling to -0.5 dB, not 0 dB.** MP3 and AAC encoders can overshoot true peaks during encoding. -0.5 dB gives safe headroom.
- The limiter should barely engage. If it's working hard (frequent, deep gain reduction), the problem is upstream — reduce makeup gain on the compressor or EQ boost levels.

---

## The Full Chain Function

```python
def process_vocal(audio, sr, config=None):
    """
    Apply the full vocal enhancement chain.

    config: dict with per-effect parameters. If None, uses defaults.
    Returns: processed numpy float32 array.
    """
    if config is None:
        config = {}

    result = audio.copy()

    # 1. Noise Gate
    if config.get('gate_enabled', True):
        result = noise_gate(result, sr, **config.get('gate', {}))

    # 2. Noise Reduction
    if config.get('noise_reduction_enabled', True):
        result = reduce_noise(result, sr, **config.get('noise_reduction', {}))

    # 3. De-Reverb
    if config.get('dereverb_enabled', True):
        result = dereverb_spectral(result, sr, **config.get('dereverb', {}))

    # 4. High-Pass Filter
    if config.get('highpass_enabled', True):
        result = highpass_filter(result, sr, **config.get('highpass', {}))

    # 5. Parametric EQ
    if config.get('eq_enabled', True):
        bands = config.get('eq_bands', [
            {'freq_hz': 250,   'gain_db': -2.5, 'q': 0.8, 'type': 'peak'},
            {'freq_hz': 3500,  'gain_db': 1.5,  'q': 0.7, 'type': 'peak'},
            {'freq_hz': 10000, 'gain_db': 1.0,  'q': 0.7, 'type': 'highshelf'},
        ])
        result = parametric_eq(result, sr, bands)

    # 6. Compressor
    if config.get('compressor_enabled', True):
        result = compressor(result, sr, **config.get('compressor', {}))

    # 7. De-Esser
    if config.get('deesser_enabled', True):
        result = de_esser(result, sr, **config.get('deesser', {}))

    # 8. Reverb
    if config.get('reverb_enabled', False):  # off by default
        reverb_cfg = config.get('reverb', {})
        if 'ir_path' in reverb_cfg:
            result = convolution_reverb(result, sr, **reverb_cfg)
        else:
            result = schroeder_reverb(result, sr, **reverb_cfg)

    # 9. Limiter (always on)
    result = limiter(result, sr, **config.get('limiter', {}))

    return result
```

## UI Preset Suggestions

For VocalForge, expose these as simple preset buttons rather than making the user tweak individual parameters:

| Preset | Description | What It Enables |
|---|---|---|
| **Raw** | No processing | Everything disabled |
| **Clean** | Basic cleanup only | Gate + Noise Reduction + HPF |
| **Enhanced** | Full chain, moderate settings | All effects, conservative defaults |
| **Bright** | Full chain with brightness boost | All effects, bright EQ preset |
| **Warm** | Full chain with warmth emphasis | All effects, warm EQ preset |
| **Custom** | User controls all parameters | All effects, sliders exposed |

The majority of use will be "Enhanced" — it should sound good without any user intervention.

---

## Dependencies

All MIT/BSD compatible:

| Library | License | Usage |
|---|---|---|
| `numpy` | BSD | Core DSP operations |
| `scipy` | BSD | Filters, STFT, convolution, interpolation |
| `noisereduce` | MIT | Spectral gating noise reduction |
| `soundfile` | BSD | Loading impulse response files |
