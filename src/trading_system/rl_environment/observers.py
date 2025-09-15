from collections import deque
from typing import List, Optional, Deque, Dict, Any
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class HermesObserver:
    """Conselho Hermes: observa OHLCV, métricas financeiras e PINN (inclui pinn_pred automaticamente)."""

    registered_name = "hermes_observer"

    def __init__(self, feed, portfolio, feature_keys: List[str], window_size: int = 20, prefill: bool = True):
        self.feed = feed
        self.portfolio = portfolio
        self.feature_keys = feature_keys
        self.window_size = int(window_size)
        self.prefill = bool(prefill)

        self._window: Deque[List[float]] = deque(maxlen=self.window_size)
        self._last_obs: Optional[np.ndarray] = None

    def reset(self) -> None:
        if hasattr(self.feed, "reset"):
            self.feed.reset()
        if hasattr(self.portfolio, "reset"):
            self.portfolio.reset()
        self._window.clear()
        self._last_obs = None

    def _safe_cast(self, value: Any) -> float:
        """Converte valores para float, tratando Timestamps e nulos."""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            return float(pd.to_datetime(value).timestamp())
        elif value is None:
            return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    def observe(self, env) -> np.ndarray:
        if not self.feed.has_next():
            if self._last_obs is None:
                return np.zeros((self.window_size, len(self.feature_keys)), dtype=np.float32)
            return self._last_obs

        data = self.feed.next()
        payload = dict(data)

        if "net_worth" not in payload:
            payload["net_worth"] = float(self.portfolio.net_worth)
        self.portfolio.on_next(payload)

        # Inclui pinn_pred se disponível
        row = [self._safe_cast(payload.get(k, 0.0)) for k in self.feature_keys]
        self._window.append(row)

        if len(self._window) < self.window_size:
            pad = self.window_size - len(self._window)
            obs = np.asarray(([row] * pad) + list(self._window), dtype=np.float32)
        else:
            obs = np.asarray(self._window, dtype=np.float32)

        self._last_obs = obs
        return obs

    @property
    def observation_space(self):
        import gymnasium as gym
        n_features = len(self.feature_keys)
        low = -np.inf * np.ones((self.window_size, n_features), dtype=np.float32)
        high = np.inf * np.ones((self.window_size, n_features), dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
