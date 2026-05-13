"""CREPE-based pitch tracking on uploaded audio (GPU when available).

If the env var CREPE_CHECKPOINT points at a fine-tuned state_dict (produced by
backend/training/finetune_crepe.py), it is loaded once and used instead of the
stock pretrained weights. Otherwise the pretrained CREPE 'full' model is used.
"""
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe
from loguru import logger

_CHECKPOINT_LOADED = False


def _maybe_load_finetuned() -> str | None:
    """If CREPE_CHECKPOINT is set & exists, bind it as torchcrepe's active model.
    Returns the checkpoint path used, or None if using pretrained weights."""
    global _CHECKPOINT_LOADED
    ckpt = os.environ.get("CREPE_CHECKPOINT")
    if not ckpt or not Path(ckpt).exists():
        return None
    if _CHECKPOINT_LOADED:
        return ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchcrepe.Crepe("full")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    torchcrepe.infer.model = model.to(device).eval()
    torchcrepe.infer.capacity = "full"
    _CHECKPOINT_LOADED = True
    logger.info(f"CREPE: using fine-tuned checkpoint {ckpt}")
    return ckpt

# Frame settings (CREPE expects 16 kHz)
SAMPLE_RATE = 16000
HOP_LENGTH = 160         # 10 ms hop @ 16 kHz  → frame index ≈ time*100
FMIN, FMAX = 50.0, 2000.0
CONFIDENCE_THRESHOLD = 0.50   # frame is "voiced" if periodicity >= this
PITCH_MEDIAN_FRAMES = 7       # median-filter the per-frame pitch before quantising (≈70 ms)
GAP_FILL_FRAMES = 12          # bridge unvoiced/short pitch dips ≤ this within one note (≈120 ms)
                              # — tremolo/strong vibrato can drop confidence for ~100 ms; a smaller
                              # value used to shatter a long held note into many tiny same-pitch fragments
                              # (because each run was too short for the anti-vibrato guard to fire).
MIN_NOTE_DURATION_SEC = 0.12  # drop fragments shorter than this (≈120 ms)
MIN_VOICED_FRAC = 0.35        # of the note span, this fraction must be confidently voiced
REARTIC_EDGE_FRAMES = 6       # a re-articulation point must be ≥ this from a note's edges (≈60 ms)
REARTIC_MIN_SPACING = 18      # ... and ≥ this many frames apart (≈180 ms) — above typical
                              # violin vibrato period (~150 ms) but below normal repeated-note spacing
ONSET_STRENGTH_PCTILE = 55    # onset-strength percentile that counts as a re-attack inside a run
RMS_VALLEY_RATIO = 0.78       # within a held note, a dip below this × the surrounding peak = re-bow
RMS_RISE_RATIO = 1.25         # ... and the level afterwards must climb back by at least this ×
# Anti-vibrato guard: if a run produces many cut candidates that are *evenly spaced*
# (period ≈ vibrato rate), it's one held note with vibrato — discard the cuts.
VIBRATO_MIN_CUTS = 3          # need at least this many cuts to even consider it vibrato
VIBRATO_SPACING_CV = 0.40     # ... and spacing coefficient-of-variation below this = regular = vibrato
VIBRATO_PERIOD_FRAMES = (12, 45)  # ≈120–450 ms, vibrato/slow-vibrato + long-note bow modulation

# Violin-onset gate — same 3-condition check as the main violin-checker's realtime
# session (SYSTEM_OVERVIEW.md §6.7). Trims leading silence / talking / metronome
# ticks before the first real bow stroke. Applied OFFLINE here too: any voiced
# frames before the gate fires are discarded so they cannot become spurious notes.
VIOLIN_PITCH_RANGE_HZ = (196.0, 3500.0)   # G3 .. B7
VIOLIN_SUSTAIN_FRAMES = 15                # ~150 ms of voiced frames
VIOLIN_PITCH_STD_SEMITONES = 0.4          # within-window pitch stability

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (beats, label) — used to snap a measured duration to the nearest written value
NOTE_VALUE_TABLE = [
    (4.0, "whole"), (3.0, "dotted-half"), (2.0, "half"),
    (1.5, "dotted-quarter"), (1.0, "quarter"),
    (0.75, "dotted-eighth"), (0.5, "eighth"),
    (0.375, "dotted-16th"), (0.25, "16th"), (0.125, "32nd"),
]


def _beats_to_note_value(beats: float) -> str:
    """Snap a beat count to the closest standard note value (log-scale nearest)."""
    import math
    if beats <= 0:
        return ""
    best = min(NOTE_VALUE_TABLE, key=lambda kv: abs(math.log2(beats) - math.log2(kv[0])))
    return best[1]


def _midi_to_note(midi: int) -> dict:
    full = NOTE_NAMES[midi % 12]
    step = full[0]
    accidental = full[1:] if len(full) > 1 else ""
    octave = midi // 12 - 1
    return {
        "midi": midi,
        "step": step,
        "accidental": accidental,
        "name": full,
        "octave": octave,
        "name_with_octave": f"{full}{octave}",
    }


def _frame_features(audio: np.ndarray, sr: int, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rms[n], onset_strength[n], backtracked_onset_frame_idxs)."""
    def _fit(x: np.ndarray) -> np.ndarray:
        if len(x) >= n_frames:
            return x[:n_frames]
        return np.pad(x, (0, n_frames - len(x)), mode="edge")

    try:
        rms = _fit(librosa.feature.rms(y=audio, frame_length=2 * HOP_LENGTH, hop_length=HOP_LENGTH)[0])
    except Exception as exc:
        logger.warning(f"rms failed: {exc}"); rms = np.ones(n_frames)
    try:
        oenv = _fit(librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH))
    except Exception as exc:
        logger.warning(f"onset_strength failed: {exc}"); oenv = np.zeros(n_frames)
    try:
        # backtrack=True snaps each onset to the local energy minimum just before the
        # attack — i.e. the actual note start. wait ≈ 130 ms keeps vibrato from firing.
        onsets = librosa.onset.onset_detect(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH, units="frames",
            backtrack=True, wait=REARTIC_MIN_SPACING,
        )
        onsets = np.asarray([int(o) for o in onsets if 0 < int(o) < n_frames], dtype=int)
    except Exception as exc:
        logger.warning(f"onset_detect failed: {exc}"); onsets = np.array([], dtype=int)
    return rms, oenv, onsets


def _split_run(a: int, b: int, rms: np.ndarray, onsets: np.ndarray) -> list[int]:
    """Re-articulation cut frames inside a constant-pitch run [a, b).

    Primary signal: backtracked onset positions that land inside the run.
    Backup: an RMS *valley* — a strict local minimum that dips below
    RMS_VALLEY_RATIO × the surrounding peaks and then climbs back by ≥ RMS_RISE_RATIO.
    A vibrato-shaped pattern of many evenly-spaced cuts is discarded.
    """
    from scipy.ndimage import uniform_filter1d
    lo, hi = a + REARTIC_EDGE_FRAMES, b - REARTIC_EDGE_FRAMES
    if hi - lo < 2:
        return []

    s = uniform_filter1d(rms[a:b].astype(float), size=3, mode="nearest")
    look = max(4, int(0.12 * SAMPLE_RATE / HOP_LENGTH))   # ±~120 ms window
    cand: list[int] = []

    # backup RMS-valley candidates
    for i in range(REARTIC_EDGE_FRAMES, (b - a) - REARTIC_EDGE_FRAMES):
        win = s[max(0, i - 2): i + 3]
        if s[i] > win.min() + 1e-9:                       # must be a strict local min
            continue
        lpeak = s[max(0, i - look): i].max() if i > 0 else s[i]
        rpeak = s[i: min(len(s), i + look)].max()
        if s[i] < RMS_VALLEY_RATIO * min(lpeak, rpeak) and rpeak > RMS_RISE_RATIO * max(s[i], 1e-9):
            cand.append(a + i)

    # primary onset candidates inside the run
    for o in onsets:
        if lo <= o <= hi:
            cand.append(int(o))

    if not cand:
        return []
    cand = sorted(set(cand))

    # enforce minimum spacing
    cuts: list[int] = []
    for c in cand:
        if not cuts or (c - cuts[-1]) >= REARTIC_MIN_SPACING:
            cuts.append(c)

    # anti-vibrato: many cuts at a regular vibrato-rate spacing → it's one held note
    if len(cuts) >= VIBRATO_MIN_CUTS:
        gaps = np.diff(cuts)
        mean_gap = float(np.mean(gaps))
        cv = float(np.std(gaps) / mean_gap) if mean_gap > 0 else 1.0
        if cv < VIBRATO_SPACING_CV and VIBRATO_PERIOD_FRAMES[0] <= mean_gap <= VIBRATO_PERIOD_FRAMES[1]:
            return []
    return cuts


def _find_violin_onset(freqs: np.ndarray, conf: np.ndarray) -> int:
    """First frame index where a stable violin tone begins.

    Three conditions must hold within a sliding VIOLIN_SUSTAIN_FRAMES window:
      • pitch sits in the violin range G3–B7
      • the window is confidently voiced
      • pitch is stable (std < VIOLIN_PITCH_STD_SEMITONES)

    Returns 0 if no clean onset is found (caller falls back to "use everything").
    """
    n = len(freqs)
    if n < VIOLIN_SUSTAIN_FRAMES:
        return 0
    fmin, fmax = VIOLIN_PITCH_RANGE_HZ
    in_range = (freqs >= fmin) & (freqs <= fmax) & np.isfinite(freqs)
    voiced = (conf >= CONFIDENCE_THRESHOLD) & in_range
    midi = librosa.hz_to_midi(np.where(voiced, freqs, np.nan))
    win = VIOLIN_SUSTAIN_FRAMES
    for i in range(n - win + 1):
        if not voiced[i:i + win].all():
            continue
        seg = midi[i:i + win]
        if np.nanstd(seg) < VIOLIN_PITCH_STD_SEMITONES:
            return i
    return 0


def _quantized_pitch(freqs: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame: (voiced mask, integer MIDI after median-smoothing). Unvoiced → midi=0."""
    from scipy.ndimage import median_filter
    midi_f = librosa.hz_to_midi(np.where(freqs > 0, freqs, np.nan))
    voiced = (conf >= CONFIDENCE_THRESHOLD) & np.isfinite(midi_f)
    # median-filter only over the voiced trace (fill gaps with nearest voiced value first)
    filled = midi_f.copy()
    if voiced.any():
        idx = np.where(voiced)[0]
        filled = np.interp(np.arange(len(midi_f)), idx, midi_f[idx])
    else:
        filled = np.zeros_like(midi_f)
    sm = median_filter(filled, size=PITCH_MEDIAN_FRAMES, mode="nearest")
    q = np.where(voiced, np.round(sm).astype(int), 0)
    return voiced, q


def _build_notes(voiced: np.ndarray, q: np.ndarray,
                 rms: np.ndarray, onsets: np.ndarray) -> list[tuple[int, int]]:
    """Group frames into [start, end) note spans.

    Step 1: maximal runs of constant quantised pitch, bridging unvoiced/other-pitch
            dips up to GAP_FILL_FRAMES (kills vibrato-induced splits).
    Step 2: inside each run, split at re-articulation points (backtracked onset or
            RMS valley) — recovers re-bowed repeated notes of the same pitch.
    """
    n = len(q)

    # Step 1 — gap-tolerant runs of constant pitch
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        pitch = q[i]
        start = i
        last_good = i
        j = i + 1
        while j < n:
            if voiced[j] and q[j] == pitch:
                last_good = j
                j += 1
            elif j - last_good <= GAP_FILL_FRAMES and not (voiced[j] and abs(q[j] - pitch) >= 2):
                j += 1
            else:
                break
        runs.append((start, last_good + 1))
        i = last_good + 1

    # Step 2 — split runs at re-articulation points
    notes: list[tuple[int, int]] = []
    for (a, b) in runs:
        prev = a
        for c in _split_run(a, b, rms, onsets):
            if c - prev >= REARTIC_EDGE_FRAMES:
                notes.append((prev, c))
                prev = c
        notes.append((prev, b))
    return notes


def _frames_to_events(audio: np.ndarray, sr: int, times: np.ndarray,
                      freqs: np.ndarray, conf: np.ndarray,
                      tempo_bpm: float | None) -> list[dict]:
    """Convert per-frame CREPE pitch into discrete note events.

    Pipeline: median-smooth + quantise pitch → gap-tolerant constant-pitch runs
    (vibrato-robust) → split at strong re-articulation onsets → filter by length /
    voiced-fraction → snap duration to a written note value if a tempo is given.
    """
    voiced, q = _quantized_pitch(freqs, conf)
    n = len(q)
    rms, _oenv, onsets = _frame_features(audio, sr, n)
    spans = _build_notes(voiced, q, rms, onsets)

    hop_sec = HOP_LENGTH / sr
    sec_per_beat = (60.0 / tempo_bpm) if tempo_bpm else None
    out: list[dict] = []

    for (a, b) in spans:
        span_voiced = voiced[a:b]
        if span_voiced.sum() == 0:
            continue
        if span_voiced.mean() < MIN_VOICED_FRAC:
            continue
        dur = (b - a) * hop_sec
        if dur < MIN_NOTE_DURATION_SEC:
            continue

        # mode of the quantised pitch over voiced frames (already smoothed)
        vq = q[a:b][span_voiced]
        vals, counts = np.unique(vq, return_counts=True)
        midi_round = int(vals[int(np.argmax(counts))])
        if midi_round <= 0:
            continue

        info = _midi_to_note(midi_round)
        info.update({
            "start_sec": round(a * hop_sec, 3),
            "duration_sec": round(dur, 3),
            "frequency": float(librosa.midi_to_hz(midi_round)),
            "confidence": float(np.mean(conf[a:b][span_voiced])),
        })
        if sec_per_beat:
            beats = dur / sec_per_beat
            info["beats"] = round(beats, 3)
            info["note_value"] = _beats_to_note_value(beats)
        else:
            info["beats"] = None
            info["note_value"] = ""
        out.append(info)

    # Smart merge: a run of ≥4 contiguous same-pitch notes with very similar
    # durations and ~zero gaps almost certainly = one held note that was
    # over-split (vibrato/tremolo). Real repeated notes have varying durations
    # and audible articulation gaps. test_rep's E4×3 has only 3 → not touched.
    return _merge_overplit_runs(out, sec_per_beat)


def _merge_overplit_runs(events: list[dict], sec_per_beat: float | None) -> list[dict]:
    """Collapse a long stretch of same-pitch fragments that *should* be one note.

    Triggers when ≥4 consecutive same-pitch events have:
      * median duration < 0.4 s   (short fragments, not real played notes), and
      * every adjacent gap < 50 ms (essentially contiguous, no articulation).
    Real played repeated notes — even fast — have audible articulation gaps,
    and groups of ≥4 truly-repeated notes at < 0.4 s each are rare in violin music.
    """
    if not events:
        return events
    out: list[dict] = []
    i = 0
    while i < len(events):
        j = i
        while j < len(events) and events[j]["midi"] == events[i]["midi"]:
            j += 1
        run = events[i:j]
        if len(run) >= 4:
            durs = [e["duration_sec"] for e in run]
            gaps = [run[k + 1]["start_sec"] - (run[k]["start_sec"] + run[k]["duration_sec"])
                    for k in range(len(run) - 1)]
            median_dur = float(np.median(durs))
            if median_dur < 0.40 and (not gaps or max(gaps) < 0.05):
                merged = dict(run[0])
                end_sec = run[-1]["start_sec"] + run[-1]["duration_sec"]
                merged["duration_sec"] = round(end_sec - run[0]["start_sec"], 3)
                merged["confidence"] = round(float(np.mean([e["confidence"] for e in run])), 3)
                if sec_per_beat:
                    merged["beats"] = round(merged["duration_sec"] / sec_per_beat, 3)
                    merged["note_value"] = _beats_to_note_value(merged["beats"])
                out.append(merged)
            else:
                out.extend(run)
        else:
            out.extend(run)
        i = j
    return out


def analyze_audio(audio_path: Path, tempo_bpm: float | None = None) -> dict:
    """Run CREPE pitch tracking. Returns trace + grouped note events."""
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("Empty audio")

    duration_sec = float(len(audio) / sr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _maybe_load_finetuned()
    logger.info(f"CREPE on {device}, {duration_sec:.1f}s @ {sr}Hz"
                f"{f', checkpoint={ckpt}' if ckpt else ' (pretrained)'}")

    audio_t = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    # Decoder choice — argmax + local refinement (NOT viterbi / global weighted-mean).
    # Violin tones have strong upper harmonics, so the CREPE probability vector is
    # multi-peak; a global weighted mean drifts between the fundamental and the 2nd
    # harmonic and lands on a wrong pitch (e.g. fundamental 369 Hz, 2nd harmonic
    # 833 Hz → weighted mean ~582 Hz). Picking the argmax bin and centroiding only
    # within ±N bins of it keeps the answer on the fundamental. This matches the
    # decision in the main violin-checker (`services/audio/.../pitch.py` — see
    # SYSTEM_OVERVIEW.md §6.2).
    pitch, periodicity = torchcrepe.predict(
        audio_t,
        sample_rate=sr,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        model="full",
        decoder=torchcrepe.decode.weighted_argmax,
        return_periodicity=True,
        device=device,
        batch_size=512,
    )

    # Smooth periodicity & pitch a bit (filters spurious blips)
    periodicity = torchcrepe.filter.median(periodicity, win_length=3)
    pitch = torchcrepe.filter.mean(pitch, win_length=3)

    pitch_np = pitch.squeeze(0).cpu().numpy()
    conf_np = periodicity.squeeze(0).cpu().numpy()
    times_np = np.arange(len(pitch_np)) * HOP_LENGTH / sr

    # Trim leading silence / non-violin noise (talking, metronome click, room tone)
    onset_frame = _find_violin_onset(pitch_np, conf_np)
    if onset_frame > 0:
        conf_np = conf_np.copy()
        conf_np[:onset_frame] = 0.0   # forces those frames "unvoiced" downstream

    note_events = _frames_to_events(audio, sr, times_np, pitch_np, conf_np, tempo_bpm)

    return {
        "duration_sec": duration_sec,
        "sample_rate": sr,
        "tempo_bpm": tempo_bpm,
        "num_frames": int(len(pitch_np)),
        "num_voiced_frames": int(((conf_np >= CONFIDENCE_THRESHOLD) & (pitch_np > 0)).sum()),
        "device": device,
        "model": "crepe-finetuned" if ckpt else "crepe-pretrained",
        "note_events": note_events,
    }