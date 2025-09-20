# src/trading_system/rl_environment/environment.py
import logging
import numpy as np
import pandas as pd
from typing import Optional
from collections import deque

import gymnasium as gym
from gymnasium import spaces

from ..data.feature_engineering import add_pinn_derived_features
from ..models.pinn_handler import PINNHandler

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        full_dataset: pd.DataFrame,
        pinn_handler: PINNHandler,
        retrain_interval: int,
        window_size: int = 20,
        starting_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001
    ):
        super().__init__()

        self.pinn_handler = pinn_handler
        self.retrain_interval = retrain_interval
        self.window_size = int(window_size)
        self.starting_balance = float(starting_balance)
        self.transaction_cost_pct = transaction_cost_pct
        self.raw_df = full_dataset.copy()
        self.max_steps = len(self.raw_df) - 1

        logger.info("Gerando predições iniciais do PINN para todo o dataset...")
        initial_pinn_preds = self.pinn_handler.predict(self.raw_df)
        
        self.df = self.raw_df.copy()
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df.reset_index(inplace=True)
            
        self.df['pinn_pred'] = initial_pinn_preds
        self.df = add_pinn_derived_features(self.df)
        logger.info("Features derivadas do PINN foram adicionadas ao dataset do ambiente.")
        
        self.feature_columns = self.df.select_dtypes(include=np.number).columns.tolist()
        self.n_features = len(self.feature_columns)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(21)
        self._action_center = (self.action_space.n - 1) // 2
        
        self._current_step = 0
        self.net_worth = 0.0
        self._position = 0.0
        self._last_position = 0.0
        self.rewards_history = deque(maxlen=self.window_size)
        self.peak_net_worth = self.starting_balance

    def _get_observation(self) -> np.ndarray:
        start_idx = max(0, self._current_step - self.window_size + 1)
        end_idx = self._current_step + 1
        obs_window = self.df.iloc[start_idx:end_idx][self.feature_columns].values
        
        if obs_window.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - obs_window.shape[0], self.n_features))
            obs_window = np.vstack((padding, obs_window))

        return np.nan_to_num(obs_window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed)
        self._current_step = 0
        self.net_worth = self.starting_balance
        self._position = 0.0
        self._last_position = 0.0
        self.rewards_history.clear()
        self.peak_net_worth = self.starting_balance
        info = { "net_worth": self.net_worth, "position": self._position }
        return self._get_observation(), info

    def step(self, action):
        if self._current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, {}
        # Treinamento do PINN em intervalos definidos
        prev_price = self.df.iloc[self._current_step]["close"]
        self._current_step += 1
        done = self._current_step >= self.max_steps

        current_price = self.df.iloc[self._current_step]["close"]

        pos = (action - self._action_center) / float(self._action_center)
        self._position = pos
        # 1. Calcular o retorno bruto do log
        log_return = np.log(current_price / prev_price) if prev_price > 0 and current_price > 0 else 0.0

        # 2. Definir a nova posição e calcular o custo de transação
        self._last_position = self._position
        self._position = (action - self._action_center) / self._action_center

        trade_volume = abs(self._position - self._last_position)
        transaction_cost = trade_volume * self.transaction_cost_pct

        # 3. Calcular a recompensa baseada no retorno da posição
        position_reward = self._position * log_return

        # 4. Calcular o componente de recompensa do Sortino Ratio
        self.rewards_history.append(position_reward)
        if len(self.rewards_history) > 1:
            negative_rewards = [r for r in self.rewards_history if r < 0]
            downside_std = np.std(negative_rewards) if negative_rewards else 0
            sortino_ratio = np.mean(self.rewards_history) / (downside_std + 1e-8)
        else:
            sortino_ratio = 0.0

        # 5. Combinar os componentes na recompensa final
        # Cálculo da Penalidade por Drawdown
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        drawdown_penalty = drawdown * 0.1 # Penalidade proporcional ao drawdown

        # A recompensa agora é o retorno da posição, menos os custos, mais um bônus por consistência (Sortino)
        reward = position_reward - transaction_cost + (sortino_ratio * 0.01) - drawdown_penalty

        # Atualiza o patrimônio com base no retorno real (sem o Sortino) e custos
        self.net_worth *= (1 + position_reward - transaction_cost)

        if self._current_step % self.retrain_interval == 0 and not done:
            start_idx = max(0, self._current_step - self.retrain_interval)
            recent_data = self.raw_df.iloc[start_idx:self._current_step]
            self.pinn_handler.fine_tune(recent_data)

        obs = self._get_observation()

        # índice seguro
        safe_idx = min(self._current_step, len(self.df) - 1)
        if "timestamp" in self.df.columns:
            timestamp = self.df["timestamp"].iloc[safe_idx]
        elif "date" in self.df.columns:
            timestamp = self.df["date"].iloc[safe_idx]
        else:
            timestamp = None

        info = {
            "net_worth": self.net_worth,
            "position": self._position,
            "timestamp": timestamp,
            "pinn_pred": self.df.loc[self._current_step, 'pinn_pred'],
            "premium": self.df.loc[self._current_step, 'premium']
        }
        return obs, float(reward), self._current_step >= len(self.df) - 1, False, info



    def render(self, mode="human"):
        render_idx = self._current_step
        date = self.df.get('date', pd.Series(None)).iloc[render_idx]
        price = self.df.loc[render_idx, 'close']
        print(
            f"Step: {self._current_step} | Date: {date} | "
            f"Price: {price: .2f} | Position: {self._position: .2f} | "
            f"Net Worth: {self.net_worth: .2f}"
        )