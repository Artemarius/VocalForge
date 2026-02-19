# Noise Reduction Analysis: VOCAL_ENHANCEMENT.md Recommendations vs Current Implementation

**Date:** 2026-02-19
**Context:** Before starting Phase 8c, audit the noise reduction pipeline against the research document's recommendations.

---

## 1. Executive Summary

The current implementation wraps the same `noisereduce` library that VOCAL_ENHANCEMENT.md recommends, but **uses it differently in several important ways**. The most significant divergence is the operating mode: the doc recommends **stationary mode** with explicit noise profiling, while the current code runs **non-stationary mode** (the library default since v3.0) because it never passes `stationary=True`. There are also missing tuning parameters and the "Record Room Tone" feature the doc calls "dramatically better" is not implemented.

### Divergences at a Glance

| Aspect | Doc Recommends | Current Code | Impact |
|--------|---------------|--------------|--------|
| **Mode** | `stationary=True` | `stationary=False` (library default) | Different algorithm entirely |
| **prop_decrease default** | 0.7 | 0.75 (effects.py) | Minor — slightly more aggressive |
| **n_std_thresh_stationary** | 1.5 | Not passed | Irrelevant if non-stationary, but should be set for stationary mode |
| **freq_mask_smooth_hz** | Library default (500) | Not passed (gets 500) | OK — using library default |
| **time_mask_smooth_ms** | Library default (50) | Not passed (gets 50) | OK — using library default |
| **use_torch** | "PyTorch backend is faster" | Not used | Performance only, RTX 3060 available |
| **Room tone capture** | "Dramatically improves quality" | Not implemented | Quality opportunity |
| **Noise profiling** | `noise_clip` from room tone | Stem-guided or first-0.5s fallback | Stem-guided is arguably better |

---

## 2. What VOCAL_ENHANCEMENT.md Recommends

### Recommended Implementation (verbatim from doc)
```python
import noisereduce as nr

def reduce_noise(audio, sr, noise_clip=None, prop_decrease=0.7, stationary=True):
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        prop_decrease=prop_decrease,
        stationary=stationary,
        n_std_thresh_stationary=1.5,
    )
```

### Key Principles
1. **Stationary mode** (`stationary=True`) — assumes constant noise profile (fan, hiss, hum)
2. **Explicit noise clip** — "record 3-5 seconds of room tone before singing" for dramatically better profiling
3. **Conservative strength** — start at `prop_decrease=0.6`, increase gradually, stay below 0.85 to avoid musical noise artifacts
4. **Fixed parameters** — `n_std_thresh_stationary=1.5` (threshold sensitivity)

### Tuning Guidance
- Below 0.6: minimal effect
- 0.6-0.8: sweet spot for singing voice
- Above 0.85: "musical noise" artifacts (tinkling, underwater sound)
- For non-stationary noise (traffic): switch to `stationary=False`

---

## 3. What We Actually Implemented

### Current `reduce_noise()` (noise_reduction.py:153-225)
```python
reduced = nr.reduce_noise(
    y=y,
    sr=sample_rate,
    y_noise=y_noise,
    prop_decrease=float(strength),
    # stationary NOT passed → defaults to False (non-stationary)
    # n_std_thresh_stationary NOT passed → defaults to 1.5
    # freq_mask_smooth_hz NOT passed → defaults to 500
    # time_mask_smooth_ms NOT passed → defaults to 50
    # use_torch NOT passed → defaults to False
)
```

### What We Added Beyond the Doc
1. **Stem-guided noise profiling** (`estimate_noise_from_stem()`) — uses the Demucs-separated vocal stem to identify silent regions in the recording, then extracts those regions as a noise reference. This is more sophisticated than the doc's "record room tone" approach.
2. **Three-strategy fallback** (`estimate_noise_profile()`) — "start", "end", or "quietest" segment
3. **NaN/inf sanitization** — protects against noisereduce producing bad values when noise bins approach zero
4. **HPF integration** — high-pass filter applied before spectral gating (when called standalone; separated in effects chain)
5. **Mono/stereo handling** — proper transposition between VocalForge's `(samples, channels)` and noisereduce's `(channels, samples)` convention

---

## 4. Deep Dive: Stationary vs Non-Stationary Mode

This is the most significant difference and deserves detailed analysis.

### Stationary Mode (`stationary=True`)
- Computes noise statistics **once** from the `y_noise` reference (or full signal if no reference)
- Applies a **fixed threshold** per frequency bin across the entire signal
- Gate decision: `magnitude > mean_noise + n_std_thresh * std_noise`
- Best for **constant** noise (AC hum, mic hiss, fan, room tone)
- **Requires good noise profiling** — garbage in, garbage out
- **Simpler, more predictable** — easier to reason about and tune

### Non-Stationary Mode (`stationary=False`, our current behavior)
- Computes noise statistics using a **sliding window** (`time_constant_s=2.0` by default)
- Threshold **adapts** over time per frequency band
- Uses a **sigmoid function** for soft masking instead of hard threshold
- Controlled by: `thresh_n_mult_nonstationary`, `sigmoid_slope_nonstationary`, `time_constant_s`
- **Does not require** a separate noise clip (but can use one for initialization)
- Better for **time-varying** noise (intermittent sounds, traffic, HVAC cycling)

### Which Is Better for Singing Voice?

| Factor | Stationary | Non-Stationary | Winner |
|--------|-----------|----------------|--------|
| Constant room noise (fan, hiss) | Excellent | Good | Stationary |
| Varying noise (traffic, HVAC cycles) | Poor | Excellent | Non-stationary |
| Vocal dynamics (quiet vs loud passages) | Risk of over-gating quiet parts | Adapts — less risky | Non-stationary |
| Predictability / tunability | Very predictable | Harder to tune | Stationary |
| Artifacts with singing | Musical noise if too aggressive | Softer artifacts but can affect sustained notes | Tie |
| Need for noise reference | Critical | Optional | Non-stationary |
| Performance | Slightly faster | Slightly slower | Stationary |

**Verdict:** Neither is strictly better. Stationary mode gives cleaner results **when the noise profile is good** (which our stem-guided profiling provides). Non-stationary mode is more forgiving when profiling is poor but can interact unpredictably with sustained vocal notes (vibrato, held notes) because the sliding window may mistake sustained vocal energy for non-stationary noise.

---

## 5. Detailed Parameter Gap Analysis

### Parameters We Should Be Passing But Aren't

#### A. `stationary` (Critical)
- **Current:** Not passed, defaults to `False`
- **Doc recommends:** `True`
- **Impact:** We're running a fundamentally different algorithm
- **Recommendation:** Add as a parameter, default `True` to match doc guidance. When the user has good noise profiling (stem-guided), stationary mode is likely better. Offer non-stationary as an option.

#### B. `n_std_thresh_stationary` (Important for stationary mode)
- **Current:** Not passed, defaults to 1.5
- **Doc recommends:** 1.5
- **Impact:** None right now (we're in non-stationary mode). Becomes relevant if we switch to stationary.
- **Recommendation:** Pass explicitly for clarity and future-proofing.

#### C. `use_torch` / `device` (Performance)
- **Current:** Not passed, defaults to `False` / `"cuda"`
- **Doc mentions:** "PyTorch backend is faster for longer recordings"
- **Impact:** Processing a 4-min song is ~3-8s on CPU vs ~0.5-1.5s on GPU
- **Recommendation:** Auto-detect CUDA availability and use torch backend when available. Low risk, straightforward improvement.

#### D. `n_fft` (Quality tuning)
- **Current:** Not passed, defaults to 1024
- **Impact:** Controls frequency resolution vs time resolution trade-off. 1024 is reasonable.
- **Recommendation:** Leave at default unless tuning reveals issues.

#### E. `freq_mask_smooth_hz` and `time_mask_smooth_ms` (Artifact control)
- **Current:** Not passed, defaults to 500 Hz / 50 ms
- **Impact:** These are the primary controls for avoiding musical noise artifacts with singing voice. Higher values = fewer artifacts but less precise NR.
- **Recommendation:** Consider slightly higher values for singing voice (e.g., 750 Hz / 75 ms). Could expose as an advanced parameter.

---

## 6. Noise Profiling Comparison

### What the Doc Recommends: Room Tone Capture
```
User presses "Record Room Tone" → 3 seconds of silence captured → stored as noise_clip
→ passed directly to nr.reduce_noise(y_noise=noise_clip)
```

**Pros:**
- Pure noise reference — no vocal content contamination
- Simple, reliable, deterministic
- User has clear mental model ("I'm recording the room")

**Cons:**
- Requires user action before every recording session
- Noise profile may drift if environment changes mid-session (e.g., AC turns on)
- Extra UX step that may annoy frequent users

### What We Implemented: Stem-Guided Profiling
```
Demucs separates song → vocal_sep stem identifies silent regions → corresponding
regions in vocal_rec extracted as noise reference → passed to noisereduce
```

**Pros:**
- Fully automatic — zero user effort
- Captures noise from multiple points throughout the recording (not just the start)
- Noise profile includes any changes that occurred during the session
- Leverages VocalForge's unique architecture (already have separated stems)

**Cons:**
- Depends on Demucs separation being available (not always — user might skip separation)
- Demucs vocal-sep may have residual signal in "silent" regions (imperfect separation)
- The -40 dB silence threshold is a heuristic that may not work for all recordings
- Falls back to "first 0.5s" heuristic when stem is unavailable — which is poor quality if the recording starts immediately

### What We Also Have: First-0.5s Fallback
Used when no stem is available. Takes the first half-second of the recording as the noise reference.

**Pros:** Zero-effort fallback
**Cons:** If the user starts singing immediately, this samples vocal signal as "noise" — catastrophic for NR quality.

### Assessment
The stem-guided approach is **better than room tone capture** when Demucs stems are available. The fallback ("first 0.5s") is **worse than room tone capture**. Both approaches can coexist.

---

## 7. Proposal: Dual-Mode Implementation

### Concept
Offer both stationary and non-stationary modes, let the user choose, and default to the optimal choice based on available data.

### Parameters to Add to `reduce_noise()`

```python
def reduce_noise(
    data, sample_rate,
    noise_clip=None,
    strength=1.0,
    guide_stem=None,
    hpf_cutoff_hz=0.0,
    # --- NEW PARAMETERS ---
    stationary=True,              # Match doc recommendation
    n_std_thresh=1.5,             # Stationary threshold sensitivity
    use_torch=False,              # GPU acceleration
    freq_smooth_hz=500,           # Frequency mask smoothing
    time_smooth_ms=50,            # Temporal mask smoothing
):
```

### Parameters to Add to Effects Chain Config

```python
"spectral_noise_reduction": {
    "enabled": True,
    "strength": 0.75,
    "guide_stem": None,
    "stationary": True,           # NEW
    "n_std_thresh": 1.5,          # NEW (only used in stationary mode)
    "use_torch": False,           # NEW (auto-detect recommended)
}
```

### UI Changes in MixPanel

The existing NR dropdown (Subtle / Moderate / Aggressive) maps to strength only. Options:

**Option A — Minimal UI change (recommended):**
- Keep Subtle/Moderate/Aggressive dropdown for strength
- Add a small "Mode" toggle: "Auto" / "Stationary" / "Adaptive"
  - "Auto" (default): uses stationary when stem/noise_clip available, non-stationary otherwise
  - "Stationary": always `stationary=True`
  - "Adaptive": always `stationary=False`

**Option B — Advanced panel:**
- Full parameter exposure in Custom preset (n_std_thresh, freq_smooth, time_smooth)
- Most users won't touch these

---

## 8. Impact on Pipeline

### What Changes
1. `noise_reduction.py`: Add `stationary`, `n_std_thresh`, `use_torch`, smoothing params to `reduce_noise()`
2. `effects.py`: Pass new params through `spectral_noise_reduction()` wrapper
3. `mix_panel.py`: Add mode selector to NR controls
4. `main_window.py`: Pass new config through workers

### What Doesn't Change
- Stem-guided profiling (stays as-is, works with both modes)
- HPF (separate stage, unchanged)
- Noise gate (stage 1, unchanged)
- Effects chain order (NR stays at position 2)
- Test structure (add tests, don't remove any)

### Risk Assessment
- **Low risk:** Adding `stationary=True` to the existing `nr.reduce_noise()` call. We're already calling the same library — just toggling a parameter.
- **Low risk:** Adding `use_torch` for GPU acceleration. Fallback is CPU (current behavior).
- **Medium risk:** Changing the default behavior from non-stationary to stationary. Users who have been getting acceptable results with the current non-stationary mode might notice a difference. Mitigated by the "Auto" mode that preserves current behavior when no noise profile is available.
- **No risk:** Passing smoothing parameters — they currently use library defaults anyway.

---

## 9. Is It Worth the Effort?

### Effort Estimate
- Code changes: ~1-2 hours (small, surgical changes to existing functions)
- UI changes: ~30 minutes (one combo box addition)
- Testing: ~1 hour (add stationary mode tests, verify both paths)
- **Total: ~3-4 hours**

### Expected Quality Improvement
- **Stationary mode with good noise profile:** Measurably cleaner output for constant noise environments (which is the typical home recording scenario — fan, AC, computer)
- **GPU acceleration:** 4-8x faster NR processing (seconds instead of seconds, but still a nice UX improvement)
- **Explicit smoothing control:** Ability to fine-tune artifact tradeoff for problem recordings
- **Correct parameter passing:** Eliminates silent behavioral dependency on library defaults that could change between noisereduce versions

### Verdict
**Yes, worth it.** The effort is small (3-4 hours), the risk is low (same library, just different parameters), and the quality improvement from matching the doc's researched recommendations is meaningful. The biggest win is aligning the default to `stationary=True` for the common case (constant room noise with good stem-guided profiling).

---

## 10. Room Tone Capture: Future Consideration

The doc's "Record Room Tone" feature is a genuinely valuable idea but has UX implications:

### Pros
- Pure noise reference without any algorithmic estimation
- Works even without Demucs separation
- Improves the fallback path significantly

### Cons
- Extra user step in the recording workflow
- Needs UI space (button, status indicator, playback of captured tone)
- Needs persistence (save room tone clip with project or per-session)

### Recommendation
Defer to Phase 11 (Settings & UX Polish). The stem-guided profiling already provides excellent noise references when Demucs is used, and that's the primary workflow. Room tone capture would mainly help the edge case where the user skips separation and records directly over an imported minus track.

---

## 11. Implementation Checklist

If proceeding with the dual-mode implementation:

- [ ] `noise_reduction.py`: Add `stationary`, `n_std_thresh`, `use_torch`, `freq_smooth_hz`, `time_smooth_ms` params to `reduce_noise()`
- [ ] `noise_reduction.py`: Pass all new params through to `nr.reduce_noise()`
- [ ] `noise_reduction.py`: Auto-detect CUDA for `use_torch` default
- [ ] `effects.py`: Add new params to `DEFAULT_CONFIG["spectral_noise_reduction"]`
- [ ] `effects.py`: Pass through in `spectral_noise_reduction()` wrapper
- [ ] `mix_panel.py`: Add NR mode selector (Auto / Stationary / Adaptive)
- [ ] `main_window.py`: Pass mode through effects config in workers
- [ ] `tests/test_noise_reduction.py`: Test stationary mode explicitly
- [ ] `tests/test_noise_reduction.py`: Test non-stationary mode explicitly
- [ ] `tests/test_noise_reduction.py`: Test `use_torch=False` path (can't test GPU in CI)
- [ ] `tests/test_effects.py`: Verify new config params propagate
- [ ] Manual A/B test: compare stationary vs non-stationary on a real SM58 recording

---

## 12. Summary of Differences

```
VOCAL_ENHANCEMENT.md says:          We do:
-------------------------------     --------------------------------
stationary=True                     stationary=False (lib default)
prop_decrease=0.7                   prop_decrease=0.75 (close)
n_std_thresh_stationary=1.5         not passed (happens to be 1.5)
noise_clip from room tone           stem-guided + first-0.5s fallback
4 parameters passed                 3 parameters passed
no GPU acceleration mentioned       no GPU acceleration used
simple wrapper                      full pipeline with HPF, stem, NaN guard
```

**Bottom line:** We're using the right library but in a different mode. The doc's recommendation of stationary mode is well-suited to our use case (constant room noise, good noise profiling via stems). We should align to it while keeping non-stationary as an option for edge cases.
