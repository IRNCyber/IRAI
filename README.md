# IRAI — Offline-First Voice Assistant

IRAI is a privacy-focused, offline-first voice assistant designed to run entirely on local hardware with zero internet dependency.

## Overview

IRAI operates without cloud services, APIs, or internet connectivity. It leverages local models for speech recognition, language understanding, and speech synthesis — making it ideal for privacy-critical environments, home automation over LAN, in-car use without cellular, and emergency backup scenarios.

## Architecture

```
Microphone → [Whisper STT] → [Local LLM (≤7B, INT4)] → [Piper TTS] → Speaker
                                      ↕
                              [Local Vector DB]
                           (calendar, PDFs, wiki snapshots)
```

## Capabilities

| Feature | Implementation |
|---|---|
| Speech-to-Text | Whisper / MMASR (local) |
| Language Model | ≤7B parameters, INT4 quantized |
| Text-to-Speech | Piper / Tortoise-TTS (local) |
| Device Control | Lights, volume, timers, alarms (LAN) |
| Knowledge Base | Offline Wikipedia (top 10k), user PDFs, cached weather |
| Utilities | Math, unit conversion, calendar (offline) |

## Project Structure

```
IRAI/
├── README.md
├── pyproject.toml              # Dependencies and project config
├── prompts/
│   ├── system_prompt.md        # Core system instructions
│   ├── user_prompt_template.md # Per-command prompt template
│   └── rag_preprompt.md        # Knowledge base RAG pre-prompt
├── src/
│   ├── main.py                 # Voice assistant loop + CLI entry point
│   ├── config.py               # YAML config loader
│   ├── stt.py                  # Speech-to-text (Whisper)
│   ├── llm.py                  # Local LLM (llama-cpp-python)
│   ├── tts.py                  # Text-to-speech (Piper)
│   ├── knowledge.py            # Offline RAG (FAISS + sentence-transformers)
│   ├── device.py               # Device control (volume, timers, weather)
│   ├── wake_word.py            # Wake word detection
│   └── prompts.py              # Prompt builder from templates
├── config/
│   ├── models.yaml             # STT / LLM / TTS model settings
│   └── device.yaml             # Audio, wake word, knowledge paths
├── knowledge/                  # Preloaded knowledge base files
└── docs/                       # Documentation and guides
```

## Use Cases

- **Home automation** — LAN-only smart home control
- **Car assistant** — No cellular required
- **Privacy-critical environments** — All data stays on-device
- **Emergency backup** — Works without connectivity

## Personality

IRAI is calm, helpful, and slightly terse. It acknowledges offline limitations upfront and uses cached data transparently.

## Getting Started

### Prerequisites

- Python 3.10+
- A GGUF-format LLM model file (e.g., Mistral 7B Q4_K_M)
- Working microphone and speakers (for voice mode)

### Installation

```bash
pip install -e .
```

### Configuration

1. Download a GGUF model and set the path in `config/models.yaml`:
   ```yaml
   llm:
     model: /path/to/your-model.Q4_K_M.gguf
   ```

2. Optionally add documents to `knowledge/user_docs/` (PDF or TXT).

3. Optionally add calendar events to `knowledge/calendar.json`:
   ```json
   [{"title": "Team standup", "date": "2025-01-20", "time": "09:00"}]
   ```

### Usage

**Voice mode** (full pipeline — mic → STT → LLM → TTS → speaker):
```bash
irai
```

**Text mode** (test without audio hardware):
```bash
irai --text "What's the weather today?"
```

**With debug logging:**
```bash
irai --verbose
```

## License

MIT
