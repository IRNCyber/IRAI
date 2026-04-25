"""IRAI — Main voice assistant loop."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from src.config import IraiConfig, load_config
from src.device import DeviceController
from src.knowledge import KnowledgeBase
from src.llm import LocalLLM
from src.prompts import PromptBuilder
from src.stt import SpeechToText
from src.tts import TextToSpeech
from src.wake_word import SimpleWakeWordDetector, WakeWordDetector

logger = logging.getLogger("irai")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class IraiAssistant:
    """Main IRAI voice assistant orchestrator."""

    def __init__(self, config: IraiConfig) -> None:
        self._config = config
        self._stt = SpeechToText(config.stt)
        self._llm = LocalLLM(config.llm)
        self._tts = TextToSpeech(config.tts)
        self._knowledge = KnowledgeBase(config.knowledge, _PROJECT_ROOT)
        self._device = DeviceController()
        self._prompts = PromptBuilder(config)
        self._wake_detector = WakeWordDetector(config.wake_word, config.audio)
        self._fallback_wake = SimpleWakeWordDetector(config.wake_word.phrase)
        self._running = False

    def load_models(self) -> None:
        """Load all ML models into memory."""
        logger.info("Loading IRAI models...")
        self._stt.load()
        self._llm.load()
        self._tts.load()
        self._knowledge.load()
        self._wake_detector.load()
        logger.info("All models loaded. IRAI is ready.")

    def _record_audio(self, duration: float = 5.0) -> np.ndarray:
        """Record audio from the microphone."""
        sample_rate = self._config.audio.sample_rate
        channels = self._config.audio.channels
        logger.debug("Recording %0.1fs of audio...", duration)
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()
        return audio.flatten()

    def _listen_for_wake_word(self) -> bool:
        """Listen continuously for the wake word."""
        chunk_duration = 2.0
        audio = self._record_audio(duration=chunk_duration)

        if self._wake_detector.detect(audio):
            return True

        text = self._stt.transcribe(audio.astype(np.float32) / 32768.0)
        if self._fallback_wake.check_transcript(text):
            return True

        return False

    def _listen_for_command(self, duration: float = 5.0) -> str:
        """Record and transcribe a user command after wake word."""
        self._tts.speak("Listening.")
        audio = self._record_audio(duration=duration)
        text = self._stt.transcribe(audio.astype(np.float32) / 32768.0)
        logger.info("User said: %s", text)
        return text

    def _process_command(self, command: str) -> str:
        """Process a voice command and generate a response."""
        device_state = self._device.get_state()

        knowledge_results = self._knowledge.search(command)
        knowledge_context = "\n".join(knowledge_results) if knowledge_results else None

        prompt = self._prompts.build(
            user_command=command,
            device_state=device_state,
            knowledge_context=knowledge_context,
        )

        response = self._llm.generate(prompt)
        return response

    def _handle_device_commands(self, command: str) -> Optional[str]:
        """Handle built-in device commands directly."""
        lower = command.lower()

        if "volume" in lower:
            if "up" in lower or "louder" in lower:
                state = self._device.get_state()
                return self._device.set_volume(state.volume + 10)
            elif "down" in lower or "quieter" in lower or "softer" in lower:
                state = self._device.get_state()
                return self._device.set_volume(state.volume - 10)

        if "weather" in lower:
            return self._device.get_weather()

        if "timer" in lower or "set a timer" in lower:
            for word in command.split():
                if word.isdigit():
                    return self._device.set_timer(int(word))
            return self._device.set_timer(5, "Quick timer")

        if "what time" in lower:
            state = self._device.get_state()
            return f"It's {state.time}."

        if "calendar" in lower or "schedule" in lower:
            return self._knowledge.get_calendar_summary()

        return None

    def process_text(self, text: str) -> str:
        """Process a text command (for testing without audio)."""
        device_response = self._handle_device_commands(text)
        if device_response:
            return device_response
        return self._process_command(text)

    def run(self) -> None:
        """Run the main voice assistant loop."""
        self._running = True
        logger.info("IRAI is listening. Say '%s' to activate.", self._config.wake_word.phrase)

        while self._running:
            try:
                if self._config.wake_word.enabled:
                    if not self._listen_for_wake_word():
                        continue

                command = self._listen_for_command()
                if not command.strip():
                    self._tts.speak("I didn't catch that.")
                    continue

                device_response = self._handle_device_commands(command)
                if device_response:
                    self._tts.speak(device_response)
                    continue

                response = self._process_command(command)
                self._tts.speak(response)

            except KeyboardInterrupt:
                break
            except Exception:
                logger.error("Error processing command.", exc_info=True)
                self._tts.speak("Something went wrong. Try again.")

        logger.info("IRAI stopped.")

    def stop(self) -> None:
        """Stop the assistant loop."""
        self._running = False


def main() -> None:
    """Entry point for IRAI."""
    parser = argparse.ArgumentParser(description="IRAI — Offline voice assistant")
    parser.add_argument(
        "--text",
        type=str,
        help="Process a single text command (no audio).",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=None,
        help="Path to models.yaml config file.",
    )
    parser.add_argument(
        "--device-config",
        type=str,
        default=None,
        help="Path to device.yaml config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    models_path = Path(args.models_config) if args.models_config else None
    device_path = Path(args.device_config) if args.device_config else None
    config = load_config(models_path=models_path, device_path=device_path)

    assistant = IraiAssistant(config)
    assistant.load_models()

    if args.text:
        response = assistant.process_text(args.text)
        print(f"IRAI: {response}")
        return

    def signal_handler(sig: int, frame: object) -> None:
        logger.info("Shutting down...")
        assistant.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    assistant.run()


if __name__ == "__main__":
    main()
