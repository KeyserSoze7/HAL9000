"""
main.py — Pipeline orchestrator
Mic → VAD → ASR → LLM → TTS → Speaker

Run with:
    python main.py
    python main.py --text-only     # skip mic, type instead
    python main.py --no-tts        # skip TTS, print response instead
"""

import argparse
import sys

from vad import load_silero_vad, wait_for_activity, record_until_silence
from asr import load_asr_model, transcribe
from llm import load_llm, generate
from tts import speak



BANNER = """
╔══════════════════════════════════════╗
║      OFFLINE VOICE ASSISTANT         ║
║   VAD · Whisper · Qwen · Piper       ║
╚══════════════════════════════════════╝
  Ctrl+C to quit
"""



# Main loop


def run_voice_loop(llm, asr_model, vad_model, use_tts: bool):
    """Full voice pipeline loop."""
    history = []
    print("\n[MAIN] Ready. Say something!\n")

    while True:
        try:
            # 1. Wait for activity (cheap gate)
            wait_for_activity(vad_model, verbose=False)

            # 2. Record full utterance
            audio = record_until_silence(vad_model, verbose=True)
            if audio is None:
                continue

            # 3. ASR
            user_text = transcribe(asr_model, audio)
            if not user_text:
                continue

            print(f"\n  You: {user_text}")

            # 4. LLM
            response, history = generate(llm, user_text, history)
            print(f"  Assistant: {response}\n")

            # 5. TTS
            if use_tts:
                speak(response)

        except KeyboardInterrupt:
            print("\n[MAIN] Goodbye.")
            sys.exit(0)


def run_text_loop(llm, use_tts: bool):
    """Text-only loop — useful for testing LLM + TTS without mic."""
    history = []
    print("\n[MAIN] Text mode. Type your message.\n")

    while True:
        try:
            user_text = input("  You: ").strip()
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit", "bye"):
                print("[MAIN] Goodbye.")
                break

            response, history = generate(llm, user_text, history)
            print(f"  Assistant: {response}\n")

            if use_tts:
                speak(response)

        except KeyboardInterrupt:
            print("\n[MAIN] Goodbye.")
            sys.exit(0)



# Entry point


def main():
    parser = argparse.ArgumentParser(description="Offline Voice Assistant")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use keyboard input instead of microphone",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Print responses instead of speaking them",
    )
    args = parser.parse_args()

    print(BANNER)
    use_tts = not args.no_tts

    # Load models
    print("[MAIN] Loading models...\n")
    llm = load_llm()

    if not args.text_only:
        asr_model = load_asr_model()
        vad_model, _ = load_silero_vad()
        run_voice_loop(llm, asr_model, vad_model, use_tts)
    else:
        run_text_loop(llm, use_tts)


if __name__ == "__main__":
    main()
