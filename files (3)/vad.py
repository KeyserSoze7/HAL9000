"""
vad.py — Silero VAD + microphone capture
Listens continuously, yields audio chunks only when speech is detected.
"""

import numpy as np
import sounddevice as sd
import torch
import queue
import threading

# --- Config ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        # seconds per VAD chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 8       # consecutive silent chunks before cutting off
MIN_SPEECH_CHUNKS = 2       # minimum chunks to consider valid speech


def load_silero_vad():
    """Load Silero VAD model from torch hub."""
    print("[VAD] Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    (get_speech_timestamps, _, _, _, _) = utils
    print("[VAD] Silero VAD loaded.")
    return model, get_speech_timestamps


def is_speech(model, audio_chunk: np.ndarray, threshold: float = 0.5) -> bool:
    """Return True if Silero detects speech in the chunk."""
    tensor = torch.from_numpy(audio_chunk).float()
    confidence = model(tensor, SAMPLE_RATE).item()
    return confidence > threshold


def record_until_silence(model, verbose: bool = True) -> np.ndarray | None:
    """
    Blocking call. Records from mic, returns a numpy float32 array
    of the full utterance once silence is detected. Returns None if
    nothing was captured.
    """
    audio_buffer = []
    silent_chunks = 0
    speech_started = False
    speech_chunk_count = 0

    if verbose:
        print("[VAD] Listening... (speak now)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
    ) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk = chunk[:, 0]  # mono

            speaking = is_speech(model, chunk)

            if speaking:
                if not speech_started and verbose:
                    print("[VAD] Speech detected.")
                speech_started = True
                speech_chunk_count += 1
                silent_chunks = 0
                audio_buffer.append(chunk)
            else:
                if speech_started:
                    silent_chunks += 1
                    audio_buffer.append(chunk)  # keep trailing silence for natural cuts
                    if silent_chunks >= SILENCE_THRESHOLD:
                        break  # utterance complete

    if speech_chunk_count < MIN_SPEECH_CHUNKS:
        if verbose:
            print("[VAD] Too short, ignoring.")
        return None

    utterance = np.concatenate(audio_buffer, axis=0)
    if verbose:
        print(f"[VAD] Captured {len(utterance) / SAMPLE_RATE:.1f}s of audio.")
    return utterance


# --- Wake word variant (optional) ---
# If you want a simple energy-based wake detector instead of keyword spotting,
# just lower the threshold below and call record_until_silence only after it fires.

def wait_for_activity(model, threshold: float = 0.6, verbose: bool = True):
    """
    Lightweight loop that blocks until any speech activity is detected.
    Use this as a cheap "wake" gate before full recording.
    """
    if verbose:
        print("[VAD] Waiting for activity...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
    ) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk = chunk[:, 0]
            if is_speech(model, chunk, threshold=threshold):
                return  # hand off to full recorder
