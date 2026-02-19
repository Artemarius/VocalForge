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
┌──────────────────────────┐
│  1. Noise Gate            │  Silence gaps between phrases
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  2. Noise Reduction       │  Remove stationary background noise (Pass 1 — main)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  3. Gain Rider            │  Auto-level loud/quiet sections
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  4. De-Plosive            │  Remove P/B/T low-freq bursts
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  5. NR Cleanup (Pass 2)   │  Light cleanup of gain-rider-amplified residual noise
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  6. De-Reverb             │  Reduce room reverb (the "well" sound)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  7. High-Pass Filter      │  Cut rumble below vocal range
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  8. Parametric EQ         │  Shape tonal balance
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  9. Compressor (Peak)     │  Fast attack, catch transient spikes
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│ 10. Compressor (Body)     │  Slow attack, smooth overall dynamics
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│ 11. De-Esser              │  Tame sibilance (sss/shh sounds)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│ 12. Soft Clipper          │  Gently round remaining peaks (saturation)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│ 13. Reverb (optional)     │  Add controlled, pleasant space
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│ 14. Limiter               │  Prevent clipping in final output
└─────────┬────────────────┘
          ▼
    Enhanced Vocal
```

### Why This Order

1. **Gate first** — silence the gaps so noise reduction doesn't waste effort on them. Cleaner input for every downstream step.
2. **Noise Reduction (Pass 1) before Gain Rider** — clean the signal at its natural noise level, where noise is quieter and easier to identify/remove. The stem-guided noise profile (from `estimate_noise_from_stem`) is estimated from the original signal level, so NR should run before the gain rider changes those levels.
3. **Gain Rider after NR Pass 1** — now it levels a cleaner signal. When it boosts quiet sections, there's less noise to amplify. This is the key insight that motivated the two-pass design: previously the gain rider ran first and boosted room noise by up to 6 dB before NR could process it.
4. **De-Plosive after Gain Rider** — gain rider may change relative levels of plosive events, so detect them after leveling. Plosive bursts are low-frequency transient spikes from P/B/T consonants; removing them early prevents downstream processors from reacting to non-musical energy.
5. **NR Cleanup (Pass 2) after Gain Rider** — catches residual noise that the gain rider amplified. This pass is deliberately gentle — just cleaning up what Pass 1 missed plus what the gain rider boosted. Uses stationary mode only, no stem-guided profile.
6. **De-reverb before EQ** — EQ after de-reverb avoids boosting reverb tails in the presence range.
7. **High-pass before parametric EQ** — removes the mud before surgical shaping, so the EQ decisions are clearer.
8. **EQ before compression** — shape the tone first, then even out the dynamics of the shaped signal. Compressing first would make EQ behave unpredictably (the compressor would react to frequencies you later remove).
9. **Serial compression (peak then body)** — first compressor catches fast transient spikes (the visible peaks in the waveform), second compressor smooths overall dynamics and adds body. Two gentle stages is far more transparent than one aggressive stage.
10. **De-esser after compression** — compression can increase sibilance (it brings up quiet sibilants to the same level as louder ones), so de-ess after.
11. **Soft clipper after de-esser** — catches any remaining peaks that the compressors missed, rounds them gently with harmonic saturation rather than hard limiting. This is the "secret sauce" that professional mixes use for that smooth, even waveform.
12. **Reverb after everything** — add space to the fully processed, clean vocal. Never apply reverb to a noisy or reverberant signal.
13. **Limiter absolutely last** — safety net to prevent digital clipping in the final output.

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

## 2. Noise Reduction — Pass 1 (Spectral Gating)

### Purpose
Removes stationary background noise (hum, hiss, fan, computer noise) that persists even during singing. Unlike the noise gate which only helps during pauses, spectral gating removes noise that coexists with the vocal signal.

This is the main noise reduction pass. It runs at the signal's natural noise level (before the gain rider changes anything), where noise is quieter and easier to identify/remove. When a stem-guided noise profile is available (from `estimate_noise_from_stem`), it is used here for dramatically better noise estimation.

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

## 3. Automatic Gain Rider (Pre-Compression Leveling)

### Purpose
The single most impactful technique for taming peak spikes. Before any compression, the gain rider automatically levels the signal by measuring RMS loudness in short windows and applying inverse gain — bringing loud sections down and quiet sections up. This produces a more uniform signal so downstream compressors don't have to work as hard and react more musically.

This is the algorithmic equivalent of what mix engineers do manually with "clip gain" or "volume automation" — riding the fader in real time. The commercial plugin "Waves Vocal Rider" does exactly this and is considered an industry standard first step.

**Important:** The gain rider runs after NR Pass 1, so it levels a clean signal. This prevents the problem where boosting quiet sections (mostly noise) forces aggressive NR settings that cause underwater artifacts.

### Algorithm
1. Compute RMS loudness in overlapping windows (200-500 ms)
2. Calculate the gain needed to bring each window's RMS to a target level
3. Clamp the gain to a reasonable range (e.g., +/-6 dB) to avoid over-boosting silence or crushing dynamics
4. Smooth the gain curve with a slow follower to avoid audible pumping
5. Apply the smoothed gain to the original signal

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `target_rms_db` | -30 to -12 | -20 | Target RMS level each window is normalized toward |
| `window_ms` | 100 to 1000 | 300 | Analysis window size. Larger = smoother, less reactive |
| `max_gain_db` | 0 to 12 | 6 | Maximum boost applied to quiet sections |
| `max_cut_db` | 0 to 12 | 6 | Maximum cut applied to loud sections |
| `smoothing_ms` | 50 to 500 | 150 | Gain transition smoothing. Prevents audible pumping |
| `silence_threshold_db` | -80 to -40 | -50 | Below this level, don't apply gain (avoids boosting silence/noise) |

### Implementation
```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

def gain_rider(audio, sr, target_rms_db=-20, window_ms=300, max_gain_db=6,
               max_cut_db=6, smoothing_ms=150, silence_threshold_db=-50):
    """
    Automatic gain riding — levels the signal before compression.

    Measures RMS in overlapping windows, computes gain to reach target RMS,
    clamps to a safe range, smooths transitions, and applies to signal.

    Runs after NR Pass 1 so it levels a clean signal.
    """
    window_samples = int(sr * window_ms / 1000)
    hop = window_samples // 4
    silence_linear = 10 ** (silence_threshold_db / 20)
    target_rms_linear = 10 ** (target_rms_db / 20)

    # Compute windowed RMS
    n_windows = (len(audio) - window_samples) // hop + 1
    rms_values = np.zeros(n_windows)
    window_centers = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * hop
        end = start + window_samples
        window = audio[start:end]
        rms = np.sqrt(np.mean(window ** 2) + 1e-10)
        rms_values[i] = rms
        window_centers[i] = start + window_samples // 2

    # Compute gain for each window
    gain_db = np.zeros(n_windows)
    for i in range(n_windows):
        if rms_values[i] < silence_linear:
            gain_db[i] = 0.0  # Don't boost silence
        else:
            current_rms_db = 20 * np.log10(rms_values[i] + 1e-10)
            desired_gain = target_rms_db - current_rms_db
            # Clamp to safe range
            gain_db[i] = np.clip(desired_gain, -max_cut_db, max_gain_db)

    # Smooth the gain curve to avoid pumping
    smooth_samples = max(1, int(smoothing_ms / (window_ms / 4)))  # in window units
    gain_db_smooth = uniform_filter1d(gain_db, size=smooth_samples)

    # Interpolate to sample-level resolution
    interp_func = interp1d(window_centers, gain_db_smooth, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    gain_per_sample = interp_func(np.arange(len(audio)))

    # Apply gain
    gain_linear = 10 ** (gain_per_sample / 20)
    return (audio * gain_linear).astype(np.float32)
```

### Tuning Tips
- **window_ms = 300** is a good starting point — it corresponds roughly to the length of a syllable, so the gain rider follows the natural phrasing without reacting to individual notes
- **max_gain_db = 6** limits how much quiet sections are boosted — going higher risks amplifying noise during breaths or quiet passages
- The **silence_threshold** is critical — without it, the gain rider would boost room noise to target level during pauses
- After gain riding, the waveform should look visually more consistent in Audacity — the tall spikes should be closer to the average level
- This technique alone can reduce the peak-to-RMS ratio by 4-8 dB, which means the compressor needs to do half as much work

---

## 4. De-Plosive Filter

### Purpose
Removes plosive bursts — the low-frequency "thump" from P, B, T consonants that the pop filter doesn't fully catch. These appear as tall spikes in the waveform, often at the start of words. Unlike a static high-pass filter (which permanently removes all low-frequency content), a de-plosive only engages when a low-frequency transient is detected, preserving the natural warmth of the voice the rest of the time.

**Runs after the Gain Rider** so that plosive detection thresholds are consistent against the leveled signal.

### Algorithm
1. Extract the low-frequency energy (below 200-300 Hz) using a bandpass filter
2. Compute the envelope of this low-frequency signal
3. Detect when the envelope exceeds a threshold (plosive event)
4. During plosive events, apply a temporary high-pass filter to reduce the burst
5. Outside plosive events, pass signal unchanged

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `plosive_freq_hz` | 100 to 300 | 200 | Frequency below which plosive energy lives |
| `threshold_db` | -40 to -10 | -25 | Detection threshold for plosive events |
| `reduction_db` | 3 to 18 | 10 | How much to attenuate plosive energy |
| `attack_ms` | 0.5 to 5 | 1 | How fast the de-plosive engages |
| `release_ms` | 10 to 100 | 30 | How fast the de-plosive disengages |

### Implementation
```python
from scipy.signal import butter, sosfiltfilt

def de_plosive(audio, sr, plosive_freq_hz=200, threshold_db=-25,
               reduction_db=10, attack_ms=1, release_ms=30):
    """
    Dynamic de-plosive filter.

    Detects low-frequency transient bursts (P, B, T consonants) and
    temporarily applies high-pass filtering only during those moments.
    Preserves vocal warmth during normal singing.
    """
    # Extract low-frequency band for detection
    sos_lp = butter(4, plosive_freq_hz, btype='low', fs=sr, output='sos')
    low_band = sosfiltfilt(sos_lp, audio)

    # Envelope of low band
    envelope = np.abs(low_band)
    smooth_samples = int(sr * 0.005)  # 5ms smoothing
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode='same')

    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-10)

    # Detect plosive events
    is_plosive = envelope_db > threshold_db

    # Create gain reduction for low band
    gain_db = np.zeros(len(audio))
    gain_db[is_plosive] = -reduction_db

    # Smooth with attack/release
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    smoothed = np.zeros_like(gain_db)
    for i in range(1, len(gain_db)):
        if gain_db[i] < smoothed[i-1]:
            smoothed[i] = attack_coeff * smoothed[i-1] + (1 - attack_coeff) * gain_db[i]
        else:
            smoothed[i] = release_coeff * smoothed[i-1] + (1 - release_coeff) * gain_db[i]

    gain_linear = 10 ** (smoothed / 20)

    # Apply gain reduction ONLY to the low band, add back the rest
    high_band = audio - low_band
    return (high_band + low_band * gain_linear).astype(np.float32)
```

### Tuning Tips
- Plosives are typically concentrated below 200 Hz with most energy around 80-150 Hz
- The SM58 has a built-in presence peak and proximity effect — close-mic singing exaggerates plosives
- If the de-plosive is removing too much of the vocal body, raise the threshold
- This is a surgical tool — it should only activate a few times per song on the hardest P/B/T consonants
- Check by looking at the waveform: tall narrow spikes at the start of words that are much louder than the singing are likely plosives

---

## 5. NR Cleanup — Pass 2 (Light Residual Noise Reduction)

### Purpose
A gentle second pass of noise reduction that catches residual noise amplified by the gain rider. After NR Pass 1 removes the bulk of the noise and the gain rider levels the signal, quiet sections that were boosted may contain amplified residual noise. This pass cleans that up without the artifacts that come from running a single aggressive NR pass.

### Key Design Differences from Pass 1

| Aspect | Pass 1 (Main NR) | Pass 2 (NR Cleanup) |
|---|---|---|
| **Position** | Before gain rider (step 2) | After gain rider (step 5) |
| **Purpose** | Remove bulk noise at natural level | Clean up gain-rider-amplified residual |
| **Aggressiveness** | Medium-Strong (prop=0.6-0.85) | Gentle (prop=0.3-0.5) |
| **Noise profile** | Stem-guided or auto-detected | Auto-detected only (stationary) |
| **guide_stem** | Yes (when available) | Never — noise floor is different after Pass 1 + gain rider |
| **Mode** | Auto (stationary or adaptive) | Always stationary |

### Why No Stem-Guided Profile for Pass 2

The `estimate_noise_from_stem()` function and `guide_stem` parameter must only apply to Pass 1. Pass 2 must NOT receive the original noise profile because:
- The noise floor after Pass 1 is different from the original noise floor
- The gain rider has changed relative levels throughout the signal
- Pass 2 should auto-detect the residual noise from the signal statistics (first 0.5s or statistical estimation)

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `strength` | 0.0 to 1.0 | 0.4 | How aggressively to reduce residual noise (maps to `prop_decrease`) |
| `n_std_thresh` | 0.5 to 3.0 | 2.5 | Conservative classification — only removes clearly-noise bins |
| `mode` | "stationary" | "stationary" | Always stationary for Pass 2 |

### Presets
| Preset | strength | n_std_thresh | Notes |
|---|---|---|---|
| **Off** | — | — | Disabled entirely (for when Pass 1 was strong enough) |
| **Light** | 0.3 | 2.5 | Barely touches signal, just catches obvious residual noise |
| **Medium** | 0.5 | 2.0 | Good balance for most recordings |
| **Strong** | 0.7 | 1.5 | More aggressive, risk of artifacts if Pass 1 was already strong |

### Implementation
```python
def nr_cleanup(data, sr, strength=0.4, mode="stationary", n_std_thresh=2.5,
               use_torch=None, freq_smooth_hz=500, time_smooth_ms=50):
    """
    Second-pass noise reduction — gentle cleanup after gain rider.

    Catches residual noise that was amplified when the gain rider boosted
    quiet sections. Uses stationary mode only, no stem-guided profile.

    IMPORTANT: This function hardcodes guide_stem=None to ensure the
    original noise profile is never used for Pass 2.
    """
    if strength == 0.0:
        return data

    from vocalforge.audio.noise_reduction import reduce_noise
    return reduce_noise(
        data, sr,
        strength=strength,
        guide_stem=None,      # Never use stem-guided profile for Pass 2
        hpf_cutoff_hz=0.0,    # HPF is handled separately in the chain
        mode=mode,
        n_std_thresh=n_std_thresh,
        use_torch=use_torch,
        freq_smooth_hz=freq_smooth_hz,
        time_smooth_ms=time_smooth_ms,
    )
```

### Tuning Tips
- **Start with Light (strength=0.3)** — this is enough for most recordings where Pass 1 did the heavy lifting
- If you hear residual noise in quiet sections between phrases, try Medium
- **Never use Strong for Pass 2 if Pass 1 was already Medium or Strong** — the combined effect will introduce underwater/musical noise artifacts
- Pass 2 is most helpful when the gain rider has `max_gain_db` set above 4 dB — that's where residual noise amplification becomes noticeable
- If Pass 1 is on Strong and the vocal sounds clean, you can disable Pass 2 entirely (Off)

---

## 6. De-Reverb

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
- **Never de-reverb to complete dryness** — a totally dry vocal sounds unnaturally close and clinical. Some residual room sound is fine, especially if you'll add controlled reverb in step 13
- Listen on headphones when tuning de-reverb — room monitors add their own reverb and mask the effect

---

## 7. High-Pass Filter (Low Cut)

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

## 8. Parametric EQ

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

## 9. Compressor — Stage 1: Peak Tamer

### Purpose
The first compressor in a serial pair. Its job is to catch fast transient spikes — the tall peaks visible in the waveform that poke above the average level. It acts almost like a limiter with a fast attack, shaving 3-6 dB off only the loudest moments. This is the main weapon against the peak spikes visible in your Audacity screenshot.

### Parameters (Peak Tamer Defaults)
| Parameter | Range | Default | Description |
|---|---|---|---|
| `threshold_db` | -20 to -6 | -12 | Set high — should only catch the tallest peaks |
| `ratio` | 6 to 20 | 8.0 | High ratio — aggressive peak control |
| `attack_ms` | 0.5 to 5 | 2 | Very fast attack to catch transients |
| `release_ms` | 30 to 150 | 80 | Medium-fast release to recover quickly |
| `makeup_db` | 0 to 6 | auto | Compensate for gain reduction |
| `knee_db` | 0 to 6 | 3 | Moderate knee for natural transition |

### How It Differs From The Body Compressor
The peak tamer has a **fast attack** (catches the spike before it passes), **high ratio** (aggressively reduces it), and **high threshold** (only engages on the loudest peaks). It should be doing **2-4 dB of gain reduction** only on the tallest peaks, and **0 dB** on normal singing. If it's compressing continuously, the threshold is too low.

---

## 10. Compressor — Stage 2: Body Smoother

### Purpose
The second compressor in the serial pair. After the peak tamer has shaved off the tallest spikes, the body smoother evens out the overall dynamics with gentle, musical compression. It adds consistency, warmth, and "presence" to the vocal — making it sound polished and professional.

### Parameters (Body Smoother Defaults)
| Parameter | Range | Default | Description |
|---|---|---|---|
| `threshold_db` | -30 to -12 | -20 | Set lower — should compress most of the singing |
| `ratio` | 1.5 to 4 | 2.5 | Gentle ratio — subtle smoothing |
| `attack_ms` | 10 to 40 | 20 | Slow attack — lets the natural attack of words through |
| `release_ms` | 100 to 400 | 200 | Slow release — smooth, non-pumping |
| `makeup_db` | 0 to 10 | auto | Compensate for gain reduction |
| `knee_db` | 3 to 12 | 8 | Soft knee for very gradual, transparent compression |

### Serial Compression: Why Two Stages

Using two compressors with moderate settings instead of one aggressive compressor produces dramatically better results:

- **Single aggressive compressor** (ratio 8:1, threshold -18 dB): catches peaks AND smooths body with the same settings. The fast attack needed for peaks kills the natural attack of normal singing. The result sounds "squashed" and lifeless.

- **Two gentle compressors** (peak: 8:1 fast @ -12 dB, body: 2.5:1 slow @ -20 dB): the peak tamer only touches the spikes. The body smoother only gently smooths the rest. Each compressor does 2-4 dB of work instead of one doing 8-12 dB. The result sounds natural and controlled.

### Implementation
Use the same `compressor()` function from the original doc, called twice with different parameters:

```python
def serial_compress(audio, sr,
                    # Stage 1: Peak Tamer
                    peak_threshold_db=-12, peak_ratio=8.0,
                    peak_attack_ms=2, peak_release_ms=80,
                    # Stage 2: Body Smoother
                    body_threshold_db=-20, body_ratio=2.5,
                    body_attack_ms=20, body_release_ms=200):
    """
    Serial compression: two compressors in series.

    Stage 1 (peak tamer): fast attack, high ratio, high threshold.
    Only catches the tallest transient spikes. 2-4 dB reduction on peaks.

    Stage 2 (body smoother): slow attack, gentle ratio, lower threshold.
    Smooths overall dynamics and adds body. 2-4 dB average reduction.
    """
    # Stage 1: catch peaks
    stage1 = compressor(audio, sr,
                        threshold_db=peak_threshold_db,
                        ratio=peak_ratio,
                        attack_ms=peak_attack_ms,
                        release_ms=peak_release_ms,
                        knee_db=3)

    # Stage 2: smooth body
    stage2 = compressor(stage1, sr,
                        threshold_db=body_threshold_db,
                        ratio=body_ratio,
                        attack_ms=body_attack_ms,
                        release_ms=body_release_ms,
                        knee_db=8)

    return stage2
```

### Parallel Compression (Advanced Alternative)

For an even more natural sound, the body smoother can be run in parallel:

```python
def serial_parallel_compress(audio, sr, parallel_mix=0.6, **kwargs):
    """
    Stage 1 (serial peak tamer) + Stage 2 (parallel body smoother).
    Preserves natural dynamics while adding body.
    """
    # Stage 1: always serial (catch peaks)
    peak_tamed = compressor(audio, sr,
                            threshold_db=kwargs.get('peak_threshold_db', -12),
                            ratio=kwargs.get('peak_ratio', 8.0),
                            attack_ms=kwargs.get('peak_attack_ms', 2),
                            release_ms=kwargs.get('peak_release_ms', 80),
                            knee_db=3)

    # Stage 2: parallel (smooth body)
    body_compressed = compressor(peak_tamed, sr,
                                threshold_db=kwargs.get('body_threshold_db', -20),
                                ratio=kwargs.get('body_ratio', 2.5),
                                attack_ms=kwargs.get('body_attack_ms', 20),
                                release_ms=kwargs.get('body_release_ms', 200),
                                knee_db=8)

    # Mix dry (peak-tamed) with compressed body
    return ((1 - parallel_mix) * peak_tamed + parallel_mix * body_compressed).astype(np.float32)
```

### Tuning Tips
- **Check gain reduction meters:** Stage 1 should show 0 dB most of the time, with occasional 3-6 dB dips on peaks. Stage 2 should show constant 1-4 dB reduction.
- If Stage 1 is compressing continuously, raise its threshold
- If the vocal still sounds uneven after both stages, lower Stage 2's threshold by 2-3 dB
- **The order matters:** peak tamer MUST come first. If the body smoother comes first, it would lower the average level, making the peak tamer's threshold harder to set

### Compressor Core Implementation (Used by Both Stages)

The `compressor()` function below is the core engine used by both Stage 1 (peak tamer) and Stage 2 (body smoother), called with different parameters each time.

---

## 11. De-Esser

### Purpose
Tames sibilance — the harsh "sss", "shh", "tch", "f" sounds that become piercing in recordings. The SM58 is not particularly bright, but EQ presence boosts (step 8) and compression (steps 9-10) can both increase sibilance. The de-esser catches it after those steps.

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

## 12. Soft Clipper (Saturation)

### Purpose
The "secret weapon" for taming remaining peaks. After serial compression and de-essing, there may still be occasional transient spikes that poke above the average level. Instead of using another compressor (which has attack time and can introduce pumping), a soft clipper reshapes the peaks using a nonlinear waveshaping function — it gently rounds them off and adds subtle harmonic warmth.

This is what gives professional mixes that smooth, "filled in" waveform you see in the separated original vocal track. The soft clipper operates instantaneously (no attack/release) and introduces pleasant even-harmonic distortion rather than harsh digital clipping.

### Algorithm
Apply a nonlinear transfer function to the signal. The classic is `tanh` (hyperbolic tangent), which passes quiet signals unchanged but smoothly compresses peaks approaching the ceiling. Other options include polynomial waveshaping and the arctangent function.

### Parameters
| Parameter | Range | Default | Description |
|---|---|---|---|
| `drive` | 1.0 to 4.0 | 1.5 | How much to push into saturation. 1.0 = no effect. Higher = more clipping |
| `ceiling_db` | -3 to 0 | -1.0 | Output ceiling. Peaks above this get rounded |
| `mode` | tanh/arctan/cubic | tanh | Waveshaping function. tanh = warmest, cubic = most transparent |

### Implementation
```python
def soft_clipper(audio, sr, drive=1.5, ceiling_db=-1.0, mode='tanh'):
    """
    Soft clipping / saturation.

    Gently rounds off peaks using a nonlinear waveshaping function.
    Adds subtle harmonic warmth while taming transients.

    drive: 1.0 = no effect. 1.5 = subtle warmth. 2.0+ = noticeable saturation.
    mode:
      - 'tanh': classic tube-like warmth (smooth, even harmonics)
      - 'arctan': slightly brighter saturation
      - 'cubic': f(x) = 1.5x - 0.5x^3, most transparent, minimal coloring
    """
    ceiling_linear = 10 ** (ceiling_db / 20)

    # Normalize to ceiling before clipping
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return audio

    # Scale so peaks are at drive level relative to ceiling
    normalized = audio * (drive / max(peak, ceiling_linear))

    if mode == 'tanh':
        # tanh soft clip — warmest, most musical
        clipped = np.tanh(normalized) / np.tanh(drive)
    elif mode == 'arctan':
        # arctan soft clip — brighter character
        clipped = np.arctan(normalized) / np.arctan(drive)
    elif mode == 'cubic':
        # Polynomial: f(x) = 1.5x - 0.5x^3, hard clip at +/-1 first
        hard_clipped = np.clip(normalized / drive, -1, 1)
        clipped = 1.5 * hard_clipped - 0.5 * hard_clipped ** 3
    else:
        clipped = normalized

    # Scale back to original level, limited to ceiling
    output = clipped * ceiling_linear

    return output.astype(np.float32)
```

### Tuning Tips
- **Start with drive = 1.2-1.5** — this is the subtle range where you get peak control without audible distortion. The effect should be invisible — you shouldn't "hear" saturation, just notice the waveform is smoother
- **drive above 2.0** introduces audible warmth/saturation — use this intentionally for a "warmer" sound, not for transparent peak control
- **tanh mode** is the most forgiving and is the standard in analog emulations. It adds mostly 2nd and 3rd harmonics (musically pleasant)
- **The cubic polynomial** (`1.5x - 0.5x^3`) is mathematically chosen so its derivative is zero at the clipping points — this means the transition from linear to clipped is completely smooth with no kink
- **Use before the limiter** — the soft clipper handles 1-3 dB of peak reduction gently. The limiter behind it only needs to catch the rare remaining overshoot
- In Audacity, A/B the waveform before and after: the peaks should be visibly shorter while the average level stays the same. This is exactly the difference you see between your recording and the professional separated vocal

### Soft Clipper vs Limiter
Why use both? They complement each other:
- **Soft clipper** (step 12): reshapes peaks musically with harmonic saturation. Works instantaneously. Adds warmth. Handles the bulk of peak reduction (1-3 dB).
- **Limiter** (step 14): hard safety net with lookahead. Prevents any signal from exceeding 0 dBFS. Should barely engage if the soft clipper did its job. No coloring, just protection.

---

## 13. Reverb (Controlled, Optional)

### Purpose
After removing the room's natural (bad) reverb in step 6, the vocal may sound unnaturally dry and close — like the singer is inside your head. Adding a small amount of controlled, pleasant reverb places the voice in a nice acoustic space while keeping it clear.

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

## 14. Limiter

### Purpose
Safety net at the end of the chain. After all the processing — especially compression makeup gain, EQ boosts, and reverb — the signal may occasionally exceed 0 dBFS (digital full scale), causing clipping distortion. The limiter catches these peaks.

### Algorithm
A limiter is just a compressor with a very high ratio (inf:1) and very fast attack:
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
    Apply the full vocal enhancement chain (14 steps).

    config: dict with per-effect parameters. If None, uses defaults.
    Returns: processed numpy float32 array.
    """
    if config is None:
        config = {}

    result = audio.copy()

    # === CLEANING PHASE ===

    # 1. Noise Gate (silence gaps first)
    if config.get('gate_enabled', True):
        result = noise_gate(result, sr, **config.get('gate', {}))

    # 2. Noise Reduction — Pass 1 (main, at natural signal level)
    if config.get('noise_reduction_enabled', True):
        result = reduce_noise(result, sr, **config.get('noise_reduction', {}))

    # === LEVELING PHASE ===

    # 3. Gain Rider (level the now-clean signal)
    if config.get('gain_rider_enabled', True):
        result = gain_rider(result, sr, **config.get('gain_rider', {}))

    # 4. De-Plosive (after leveling, so detection thresholds are consistent)
    if config.get('deplosive_enabled', True):
        result = de_plosive(result, sr, **config.get('deplosive', {}))

    # 5. Noise Reduction — Pass 2 (light cleanup of gain-rider-amplified residual)
    if config.get('nr_cleanup_enabled', False):
        nr2_cfg = config.get('nr_cleanup', {
            'strength': 0.4,
            'n_std_thresh': 2.5,
            'mode': 'stationary',
        })
        result = nr_cleanup(result, sr, **nr2_cfg)

    # === TONAL SHAPING PHASE ===

    # 6. De-Reverb
    if config.get('dereverb_enabled', True):
        result = dereverb_spectral(result, sr, **config.get('dereverb', {}))

    # 7. High-Pass Filter
    if config.get('highpass_enabled', True):
        result = highpass_filter(result, sr, **config.get('highpass', {}))

    # 8. Parametric EQ
    if config.get('eq_enabled', True):
        bands = config.get('eq_bands', [
            {'freq_hz': 250,   'gain_db': -2.5, 'q': 0.8, 'type': 'peak'},
            {'freq_hz': 3500,  'gain_db': 1.5,  'q': 0.7, 'type': 'peak'},
            {'freq_hz': 10000, 'gain_db': 1.0,  'q': 0.7, 'type': 'highshelf'},
        ])
        result = parametric_eq(result, sr, bands)

    # === DYNAMICS PHASE ===

    # 9. Compressor Stage 1: Peak Tamer
    if config.get('peak_compressor_enabled', True):
        peak_cfg = config.get('peak_compressor', {})
        result = compressor(result, sr,
                            threshold_db=peak_cfg.get('threshold_db', -12),
                            ratio=peak_cfg.get('ratio', 8.0),
                            attack_ms=peak_cfg.get('attack_ms', 2),
                            release_ms=peak_cfg.get('release_ms', 80),
                            knee_db=peak_cfg.get('knee_db', 3))

    # 10. Compressor Stage 2: Body Smoother
    if config.get('body_compressor_enabled', True):
        body_cfg = config.get('body_compressor', {})
        result = compressor(result, sr,
                            threshold_db=body_cfg.get('threshold_db', -20),
                            ratio=body_cfg.get('ratio', 2.5),
                            attack_ms=body_cfg.get('attack_ms', 20),
                            release_ms=body_cfg.get('release_ms', 200),
                            knee_db=body_cfg.get('knee_db', 8))

    # 11. De-Esser
    if config.get('deesser_enabled', True):
        result = de_esser(result, sr, **config.get('deesser', {}))

    # 12. Soft Clipper
    if config.get('soft_clipper_enabled', True):
        result = soft_clipper(result, sr, **config.get('soft_clipper', {}))

    # === FINAL PHASE ===

    # 13. Reverb (optional)
    if config.get('reverb_enabled', False):
        reverb_cfg = config.get('reverb', {})
        if 'ir_path' in reverb_cfg:
            result = convolution_reverb(result, sr, **reverb_cfg)
        else:
            result = schroeder_reverb(result, sr, **reverb_cfg)

    # 14. Limiter (always on)
    result = limiter(result, sr, **config.get('limiter', {}))

    return result
```

## UI Preset Suggestions

For VocalForge, expose these as simple preset buttons rather than making the user tweak individual parameters:

| Preset | Description | What It Enables |
|---|---|---|
| **Raw** | No processing | Everything disabled |
| **Clean** | Basic cleanup only | Gate + NR Pass 1 (Medium) + HPF + Limiter |
| **Enhanced** | Full chain, moderate settings | All effects, NR Pass 2 (Light), conservative defaults |
| **Broadcast** | Maximum consistency, radio-like | All effects, NR Pass 2 (Medium), aggressive gain rider, tight serial compression |
| **Bright** | Full chain with brightness boost | All effects, NR Pass 2 (Light), bright EQ preset |
| **Warm** | Full chain with warmth + saturation | All effects, NR Pass 2 (Light), warm EQ, soft clipper drive=1.8 |
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
