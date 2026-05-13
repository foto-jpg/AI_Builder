"""Load CREPE (pretrained or fine-tuned) and run pitch tracking on an audio file.

This is the standalone "load model + inference" script for the report / demo.
It mirrors what the FastAPI backend does in app/services/audio_engine.py, but
with no web framework so it's easy to read.

Usage
-----
    # pretrained CREPE
    backend/venv/bin/python backend/training/infer.py recording.wav

    # fine-tuned checkpoint
    backend/venv/bin/python backend/training/infer.py recording.wav \
        --checkpoint backend/checkpoints/crepe_violin.pt --tempo 100
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
HOP_LENGTH = 160          # 10 ms
FMIN, FMAX = 50.0, 2000.0
CONF_THRESH = 0.7
MIN_NOTE_SEC = 0.05
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_crepe(checkpoint: str | None, device: str):
    """Return a ready-to-use torchcrepe.Crepe model.

    * checkpoint is None  → the pretrained 'full' weights shipped with torchcrepe.
    * checkpoint is a path → a fine-tuned state_dict produced by finetune_crepe.py.
    """
    import torch
    import torchcrepe
    model = torchcrepe.Crepe("full")
    if checkpoint:
        sd = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"[infer] loaded fine-tuned checkpoint: {checkpoint}")
    else:
        weights = os.path.join(os.path.dirname(torchcrepe.__file__), "assets", "full.pth")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        print("[infer] loaded pretrained CREPE 'full'")
    return model.to(device).eval()


def predict_pitch(model, audio: np.ndarray, sr: int, device: str):
    """Run CREPE over `audio`; return (pitch_hz[T], periodicity[T], times_sec[T])."""
    import torch
    import torchcrepe
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # [1, T]
    # torchcrepe.predict takes the *model* implicitly via torchcrepe.infer.model;
    # to use our own model object we feed frames through it the same way.
    # Easiest: temporarily bind our model.
    torchcrepe.infer.model = model
    torchcrepe.infer.capacity = "full"
    pitch, periodicity = torchcrepe.predict(
        audio_t, sample_rate=sr, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX, model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True, device=device, batch_size=512,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    pitch = torchcrepe.filter.mean(pitch, 3)
    pitch_np = pitch.squeeze(0).cpu().numpy()
    conf_np = periodicity.squeeze(0).cpu().numpy()
    times = np.arange(len(pitch_np)) * HOP_LENGTH / sr
    return pitch_np, conf_np, times


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def group_notes(pitch_hz, conf, times, tempo_bpm: float | None):
    """Collapse stable-pitch runs into note events (same logic as the backend)."""
    events, cur = [], None
    for t, f, c in zip(times, pitch_hz, conf):
        voiced = c >= CONF_THRESH and f > 0 and not np.isnan(f)
        if not voiced:
            if cur: events.append(cur); cur = None
            continue
        m = hz_to_midi(f)
        if cur and abs(m - cur["m_avg"]) <= 0.5:
            cur.update(end=t, n=cur["n"] + 1, m_sum=cur["m_sum"] + m,
                       m_avg=(cur["m_sum"] + m) / (cur["n"] + 1), c_sum=cur["c_sum"] + c)
        else:
            if cur: events.append(cur)
            cur = dict(start=t, end=t, m_avg=m, m_sum=m, n=1, c_sum=c)
    if cur: events.append(cur)

    spb = (60.0 / tempo_bpm) if tempo_bpm else None
    out = []
    for e in events:
        dur = e["end"] - e["start"]
        if dur < MIN_NOTE_SEC: continue
        midi = int(round(e["m_avg"]))
        full = NOTE_NAMES[midi % 12]
        rec = dict(start_sec=round(e["start"], 3), duration_sec=round(dur, 3),
                   midi=midi, name_with_octave=f"{full}{midi // 12 - 1}",
                   confidence=round(e["c_sum"] / e["n"], 3))
        if spb: rec["beats"] = round(dur / spb, 3)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description="CREPE pitch tracking on an audio file")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--checkpoint", type=str, default=None, help="fine-tuned .pt (omit for pretrained)")
    ap.add_argument("--tempo", type=float, default=None, help="BPM, for note-value column")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import librosa
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model = load_crepe(args.checkpoint, device)
    audio, sr = librosa.load(str(args.audio), sr=SAMPLE_RATE, mono=True)
    pitch_hz, conf, times = predict_pitch(model, audio, sr, device)
    events = group_notes(pitch_hz, conf, times, args.tempo)

    print(f"\n{args.audio.name}: {len(audio) / sr:.1f}s, {len(events)} note events "
          f"(device={device})\n")
    for e in events:
        beats = f"  {e['beats']:.2f}♩" if "beats" in e else ""
        print(f"  {e['start_sec']:6.2f}s  {e['name_with_octave']:>4}  "
              f"{e['duration_sec']:.2f}s  conf={e['confidence']:.0%}{beats}")


if __name__ == "__main__":
    main()