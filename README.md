# 🎙️ Offline Voice Assistant

A clean, minimal, fully offline voice assistant pipeline running entirely on CPU.

```
Mic → Silero VAD → Faster-Whisper → Qwen (llama.cpp) → Piper TTS → Speaker
```

---

## Stack

| Stage | Tool |
|---|---|
| Wake / VAD | Silero VAD (via torch hub) |
| ASR | Faster-Whisper `tiny.en` |
| LLM | Qwen 2.5 3B/4B GGUF via llama-cpp-python |
| TTS | Piper TTS |
| Function calling | Custom Python dispatcher (`tools.py`) |

**RAM usage:** ~3–4 GB total. Comfortable on 8 GB+.

---

## Setup

### 1. Clone & install Python deps

```bash
git clone https://github.com/KeyserSoze7/HAL9000`
cd voice-assistant
pip install -r requirements.txt
```

### 2. Install llama-cpp-python (CPU only)

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Place your GGUF model

```bash
mkdir models
# Copy your existing GGUF file here, e.g.:
cp ~/path/to/qwen2.5-3b-instruct-q4_k_m.gguf models/
```

Then update `MODEL_PATH` in `llm.py` to match the filename.

### 4. Install Piper TTS

Download the Piper binary for your OS from:
https://github.com/rhasspy/piper/releases

Then download a voice model (recommended: `en_US-lessac-medium`):
https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium

Place both `.onnx` and `.onnx.json` files in `./models/`.

Update `VOICE_MODEL` in `tts.py` if using a different voice.

### 5. (Linux) Install audio dependencies

```bash
sudo apt install portaudio19-dev libsndfile1
```

---

## Running

```bash
# Full voice pipeline
python main.py

# Text-only (test LLM + TTS without mic)
python main.py --text-only

# No TTS (print responses only)
python main.py --no-tts

# Both flags (pure text in/out — fastest for debugging)
python main.py --text-only --no-tts
```

---

## Testing individual modules

```bash
# Test ASR only (records 5 seconds)
python asr.py

# Test TTS only
python tts.py

# Test LLM only (text chat loop)
python llm.py
```

---

## Project structure

```
voice-assistant/
├── main.py          # Pipeline orchestrator
├── vad.py           # Silero VAD + mic capture
├── asr.py           # Faster-Whisper ASR
├── llm.py           # llama.cpp / Qwen wrapper
├── tts.py           # Piper TTS wrapper
├── tools.py         # Function-calling tools
├── requirements.txt
├── models/          # Put GGUF + Piper .onnx here
│   ├── qwen2.5-3b-instruct-q4_k_m.gguf
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
└── README.md
```

---

## Adding new tools

Open `tools.py` and:

1. Write a function that takes `args: list` and returns a `str`
2. Add it to the `TOOLS` dict
3. Add a one-line description to `TOOL_DESCRIPTIONS`

The LLM will automatically learn to call it from the system prompt. No framework needed.

---

## Troubleshooting

**No audio input / PortAudio error**
```bash
# List available devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```
Set `sd.default.device` in `vad.py` if needed.

**VAD not triggering**
Lower `threshold` in `wait_for_activity()` from `0.6` → `0.4`.

**Piper not found**
Make sure the `piper` binary is on your PATH, or set `PIPER_BINARY` in `tts.py` to its full path.

**LLM too slow**
Increase `n_threads` in `llm.py` to match your CPU core count.
