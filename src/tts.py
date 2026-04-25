"""Text-to-speech module using Piper (offline)."""

from __future__ import annotations

import io
import logging
import wave
from typing import Optional

import numpy as np

from src.config import TtsConfig

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Local text-to-speech using Piper."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._voice = None

    def load(self) -> None:
        """Load the Piper voice model."""
        from piper import PiperVoice

        logger.info("Loading Piper voice: %s", self._config.voice)
        self._voice = PiperVoice.load(self._config.voice)
        logger.info("Piper voice loaded.")

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio array.

        Args:
            text: Text to speak.

        Returns:
            Int16 audio array at the configured sample rate.
        """
        if self._voice is None:
            raise RuntimeError("TTS voice not loaded. Call load() first.")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self._voice.synthesize(text, wav)

        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

        return audio

    def speak(self, text: str) -> None:
        """Synthesize and play audio through speakers."""
        import sounddevice as sd

        audio = self.synthesize(text)
        sd.play(audio, samplerate=self._config.sample_rate, blocking=True)
        logger.debug("Spoke: %s", text)
