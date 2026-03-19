"""
tts.py — Piper TTS wrapper
Converts text to speech and plays it via sounddevice.
"""

import subprocess
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import shutil

# --- Config ---
# Path to the piper binary. If installed globally: just "piper"
PIPER_BINARY = "piper"

# Path to the .onnx voice model + its .json config
# Download from: https://github.com/rhasspy/piper/releases
# Recommended: en_US-lessac-medium.onnx (~65MB) or en_US-ryan-low.onnx (~30MB)
VOICE_MODEL = "./models/en_US-lessac-medium.onnx"
VOICE_CONFIG = "./models/en_US-lessac-medium.onnx.json"  # auto-detected if alongside .onnx


def _piper_available() -> bool:
    return shutil.which(PIPER_BINARY) is not None


def speak(text: str, verbose: bool = True) -> None:
    """
    Synthesise text with Piper and play it immediately.
    Falls back to printing if Piper isn't found.
    """
    if not text.strip():
        return

    if not _piper_available():
        print(f"[TTS] Piper not found. Response: {text}")
        return

    if verbose:
        print(f"[TTS] Speaking: {text}")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # Call piper: echo text | piper --model ... --output_file ...
        result = subprocess.run(
            [
                PIPER_BINARY,
                "--model", VOICE_MODEL,
                "--output_file", tmp_path,
            ],
            input=text.encode("utf-8"),
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"[TTS] Piper error: {result.stderr.decode()}")
            return

        # Play the WAV
        data, samplerate = sf.read(tmp_path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def speak_streaming(text: str, verbose: bool = True) -> None:
    """
    Sentence-by-sentence streaming speak.
    Slightly reduces perceived latency for long responses.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if sentence.strip():
            speak(sentence.strip(), verbose=verbose)


# --- Quick standalone test ---
if __name__ == "__main__":
    speak("Hello! I am your offline voice assistant. How can I help you today?")
