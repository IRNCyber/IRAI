"""Local LLM inference module using llama-cpp-python."""

from __future__ import annotations

import logging
from typing import Optional

from src.config import LlmConfig

logger = logging.getLogger(__name__)


class LocalLLM:
    """Offline LLM inference via llama.cpp."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._model = None

    def load(self) -> None:
        """Load the GGUF model into memory."""
        from llama_cpp import Llama

        if not self._config.model:
            raise ValueError(
                "No LLM model path configured. "
                "Set llm.model in config/models.yaml to a .gguf file path."
            )

        logger.info("Loading LLM: %s", self._config.model)
        n_gpu_layers = -1 if self._config.device == "cuda" else 0
        self._model = Llama(
            model_path=self._config.model,
            n_ctx=self._config.context_length,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("LLM loaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a response from the local LLM.

        Args:
            prompt: Full prompt string (system + user context).
            max_tokens: Override max output tokens.
            temperature: Sampling temperature.
            stop: Stop sequences.

        Returns:
            Generated text response.
        """
        if self._model is None:
            raise RuntimeError("LLM not loaded. Call load() first.")

        result = self._model(
            prompt,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature,
            stop=stop or ["USER:", "\n\n\n"],
            echo=False,
        )

        text = result["choices"][0]["text"].strip()
        logger.debug("LLM response: %s", text)
        return text
