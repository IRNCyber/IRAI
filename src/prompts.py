"""Prompt builder for IRAI — loads and formats prompt templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import IraiConfig
from src.device import DeviceState


class PromptBuilder:
    """Builds LLM prompts from local template files."""

    def __init__(self, config: IraiConfig) -> None:
        self._prompts_dir = config.prompts_dir
        self._system_prompt = self._load_prompt("system_prompt.md")
        self._user_template = self._load_prompt("user_prompt_template.md")
        self._rag_preprompt = self._load_prompt("rag_preprompt.md")

    def _load_prompt(self, filename: str) -> str:
        path = self._prompts_dir / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def build(
        self,
        user_command: str,
        device_state: DeviceState,
        knowledge_context: Optional[str] = None,
        cached_data_age_hours: int = 0,
    ) -> str:
        """Build the full prompt for the LLM.

        Args:
            user_command: Transcribed user speech.
            device_state: Current device state.
            knowledge_context: Relevant knowledge base results.
            cached_data_age_hours: Hours since last data sync.

        Returns:
            Formatted prompt string.
        """
        parts = []

        parts.append("### SYSTEM INSTRUCTIONS ###")
        parts.append(self._system_prompt)

        if knowledge_context:
            parts.append("\n### KNOWLEDGE BASE CONTEXT ###")
            parts.append(self._rag_preprompt)
            parts.append(f"\nRelevant information from local database:\n{knowledge_context}")

        parts.append("\n### USER COMMAND ###")
        parts.append(
            f"Command (spoken by user): {user_command}\n\n"
            f"Current context:\n"
            f"- Time: {device_state.time}\n"
            f"- Date: {device_state.date}\n"
            f"- Cached data age: {cached_data_age_hours} hours old\n"
            f"- Volume: {device_state.volume}%\n"
        )

        parts.append(
            "Respond as IRAI (offline assistant). "
            "Do not ask for internet. Be brief. Use local knowledge only.\n\n"
            "IRAI:"
        )

        return "\n".join(parts)
