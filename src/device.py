"""Device state and control for IRAI (LAN-only)."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "cache"


@dataclass
class DeviceState:
    """Current device state snapshot."""

    volume: int = 50
    battery: Optional[int] = None
    time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def summary(self) -> str:
        parts = [f"Time: {self.time}", f"Date: {self.date}"]
        if self.battery is not None:
            parts.append(f"Battery: {self.battery}%")
        parts.append(f"Volume: {self.volume}%")
        return ", ".join(parts)


class DeviceController:
    """Controls local device functions (volume, timers, cached data)."""

    def __init__(self) -> None:
        self._state = DeviceState()
        self._timers: list[dict] = []
        self._cached_weather: Optional[dict] = None
        self._load_cached_weather()

    def _load_cached_weather(self) -> None:
        weather_file = _CACHE_DIR / "weather.json"
        if weather_file.exists():
            try:
                with open(weather_file) as f:
                    self._cached_weather = json.load(f)
            except Exception:
                logger.warning("Failed to load cached weather.", exc_info=True)

    def get_state(self) -> DeviceState:
        """Refresh and return current device state."""
        self._state.time = datetime.now().strftime("%H:%M")
        self._state.date = datetime.now().strftime("%Y-%m-%d")
        return self._state

    def set_volume(self, level: int) -> str:
        """Set volume (0-100)."""
        level = max(0, min(100, level))
        self._state.volume = level
        try:
            subprocess.run(
                ["amixer", "set", "Master", f"{level}%"],
                capture_output=True,
                timeout=5,
            )
        except FileNotFoundError:
            logger.debug("amixer not available; volume tracked in state only.")
        return f"Volume set to {level} percent."

    def get_weather(self) -> str:
        """Return last cached weather forecast."""
        if not self._cached_weather:
            return "No cached weather data available."

        temp = self._cached_weather.get("temperature", "unknown")
        condition = self._cached_weather.get("condition", "unknown")
        cached_at = self._cached_weather.get("cached_at", "unknown time")
        return (
            f"Last cached forecast from {cached_at} shows "
            f"{condition}, {temp} degrees."
        )

    def set_timer(self, minutes: int, label: str = "Timer") -> str:
        """Set a countdown timer."""
        timer = {
            "label": label,
            "minutes": minutes,
            "set_at": datetime.now().isoformat(),
        }
        self._timers.append(timer)
        return f"{label} set for {minutes} minutes."

    def get_timers(self) -> str:
        """List active timers."""
        if not self._timers:
            return "No active timers."
        lines = []
        for t in self._timers:
            lines.append(f"- {t['label']}: {t['minutes']} min (set at {t['set_at']})")
        return "Active timers:\n" + "\n".join(lines)
