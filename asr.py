"""
asr.py — Faster-Whisper ASR wrapper
Transcribes a numpy audio array to text.
"""

import numpy as np
from faster_whisper import WhisperModel

# --- Config ---
MODEL_SIZE = "tiny.en"      # tiny.en ~150MB, fast on CPU, English only
COMPUTE_TYPE = "int8"       # int8 quantisation — fastest on CPU
DEVICE = "cpu"


def load_asr_model(model_size: str = MODEL_SIZE) -> WhisperModel:
    """Load and return the Faster-Whisper model."""
    print(f"[ASR] Loading Faster-Whisper '{model_size}' on CPU...")
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("[ASR] Model loaded.")
    return model


def transcribe(model: WhisperModel, audio: np.ndarray, verbose: bool = True) -> str:
    """
    Transcribe a float32 numpy array (16kHz mono) to a string.
    Returns empty string if nothing was recognised.
    """
    segments, info = model.transcribe(
        audio,
        beam_size=1,            # greedy — fastest
        language="en",
        vad_filter=False,       # VAD already handled upstream
        condition_on_previous_text=False,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()

    if verbose and text:
        print(f"[ASR] Transcript: {text}")
    elif verbose:
        print("[ASR] No speech recognised.")

    return text


# --- Quick standalone test ---
if __name__ == "__main__":
    import sounddevice as sd

    SAMPLE_RATE = 16000
    DURATION = 5

    print(f"Recording {DURATION}s — speak now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio[:, 0]

    asr = load_asr_model()
    result = transcribe(asr, audio)
    print(f"Result: '{result}'")
