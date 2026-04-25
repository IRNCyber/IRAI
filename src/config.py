"""Configuration loader for IRAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SttConfig:
    engine: str = "whisper"
    model: str = "base.en"
    language: str = "en"
    device: str = "cpu"


@dataclass
class LlmConfig:
    engine: str = "llama.cpp"
    model: Optional[str] = None
    quantization: str = "Q4_K_M"
    max_tokens: int = 150
    context_length: int = 2048
    device: str = "cpu"


@dataclass
class TtsConfig:
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"
    sample_rate: int = 22050
    device: str = "cpu"


@dataclass
class AudioConfig:
    input_device: str = "default"
    output_device: str = "default"
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class WakeWordConfig:
    enabled: bool = True
    phrase: str = "hey irai"


@dataclass
class KnowledgeConfig:
    wiki_snapshot: str = "knowledge/wiki_top10k.db"
    user_docs: str = "knowledge/user_docs/"
    calendar: str = "knowledge/calendar.json"


@dataclass
class CacheConfig:
    weather_max_age_hours: int = 24
    sports_max_age_hours: int = 168


@dataclass
class IraiConfig:
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    prompts_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "prompts")


def _dict_to_dataclass(cls: type, data: dict) -> object:
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(
    models_path: Optional[Path] = None,
    device_path: Optional[Path] = None,
) -> IraiConfig:
    """Load configuration from YAML files.

    Falls back to defaults if files are missing.
    """
    config = IraiConfig()

    models_path = models_path or _PROJECT_ROOT / "config" / "models.yaml"
    device_path = device_path or _PROJECT_ROOT / "config" / "device.yaml"

    if models_path.exists():
        with open(models_path) as f:
            data = yaml.safe_load(f) or {}
        if "stt" in data:
            config.stt = _dict_to_dataclass(SttConfig, data["stt"])
        if "llm" in data:
            config.llm = _dict_to_dataclass(LlmConfig, data["llm"])
        if "tts" in data:
            config.tts = _dict_to_dataclass(TtsConfig, data["tts"])

    if device_path.exists():
        with open(device_path) as f:
            data = yaml.safe_load(f) or {}
        if "audio" in data:
            config.audio = _dict_to_dataclass(AudioConfig, data["audio"])
        if "wake_word" in data:
            config.wake_word = _dict_to_dataclass(WakeWordConfig, data["wake_word"])
        if "knowledge" in data:
            config.knowledge = _dict_to_dataclass(KnowledgeConfig, data["knowledge"])
        if "cache" in data:
            config.cache = _dict_to_dataclass(CacheConfig, data["cache"])

    return config
