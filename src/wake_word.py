"""Wake word detection for IRAI."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.config import AudioConfig, WakeWordConfig

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects the wake word ("hey irai") using openwakeword."""

    def __init__(
        self,
        wake_config: WakeWordConfig,
        audio_config: AudioConfig,
    ) -> None:
        self._wake_config = wake_config
        self._audio_config = audio_config
        self._model = None

    def load(self) -> None:
        """Load the wake word model."""
        if not self._wake_config.enabled:
            logger.info("Wake word detection disabled.")
            return

        try:
            from openwakeword.model import Model

            self._model = Model(inference_framework="onnx")
            logger.info(
                "Wake word detector loaded for phrase: %s",
                self._wake_config.phrase,
            )
        except Exception:
            logger.warning(
                "openwakeword not available. Wake word detection disabled.",
                exc_info=True,
            )

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """Check if the wake word is present in an audio chunk.

        Args:
            audio_chunk: Int16 audio at 16kHz, mono.

        Returns:
            True if wake word detected.
        """
        if self._model is None:
            return False

        prediction = self._model.predict(audio_chunk)
        for model_name, score in prediction.items():
            if score > 0.5:
                logger.debug("Wake word detected (model=%s, score=%.2f)", model_name, score)
                return True
        return False

    def reset(self) -> None:
        """Reset the detector state between activations."""
        if self._model is not None:
            self._model.reset()


class SimpleWakeWordDetector:
    """Fallback wake word detector using STT transcription matching."""

    def __init__(self, phrase: str) -> None:
        self._phrase = phrase.lower().strip()

    def check_transcript(self, text: str) -> bool:
        """Check if transcribed text contains the wake phrase."""
        return self._phrase in text.lower()
