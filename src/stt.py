"""Speech-to-text module using Whisper (offline)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.config import SttConfig

logger = logging.getLogger(__name__)


class SpeechToText:
    """Local speech-to-text using OpenAI Whisper."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._model = None

    def load(self) -> None:
        """Load the Whisper model into memory."""
        import whisper

        logger.info("Loading Whisper model: %s", self._config.model)
        self._model = whisper.load_model(
            self._config.model,
            device=self._config.device,
        )
        logger.info("Whisper model loaded.")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio array to text.

        Args:
            audio: Float32 audio array, mono, 16kHz sample rate.
            language: Override language (defaults to config).

        Returns:
            Transcribed text string.
        """
        if self._model is None:
            raise RuntimeError("STT model not loaded. Call load() first.")

        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        result = self._model.transcribe(
            audio,
            language=language or self._config.language,
            fp16=(self._config.device == "cuda"),
        )
        text = result.get("text", "").strip()
        logger.debug("Transcribed: %s", text)
        return text
