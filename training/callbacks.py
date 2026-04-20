"""Custom Stable-Baselines3 callbacks."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """Logs custom city metrics from episode info when Monitor reports episode end."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "episode" in info:
                self.logger.record("city/population", float(info.get("population", 0.0)))
                self.logger.record("city/pollution", float(info.get("pollution", 0.0)))
                self.logger.record("city/traffic", float(info.get("traffic", 0.0)))
                self.logger.record("city/energy_balance", float(info.get("energy_balance", 0.0)))
        return True
