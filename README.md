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
├── prompts/
│   ├── system_prompt.md        # Core system instructions
│   ├── user_prompt_template.md # Per-command prompt template
│   └── rag_preprompt.md        # Knowledge base RAG pre-prompt
├── src/                        # Application source code
├── config/                     # Model and device configuration
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

> Documentation and setup guides coming soon. See `prompts/` for system prompt configuration.

## License

MIT
