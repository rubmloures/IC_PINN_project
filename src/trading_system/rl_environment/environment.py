# src/trading_system/rl_environment/environment.py
import logging
import numpy as np
import pandas as pd
import torch
import os
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from trading_system.data.feature_engineering import add_features, add_pinn_derived_features
from trading_system.models.PINN_model import EuropeanCallPINN, load_scaling_factors

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: pd.DataFrame, window_size: int = 20, starting_balance: float = 10000.0):
        super().__init__()
        self.df = dataset.copy().reset_index(drop=True)
        self.window_size = int(window_size)

        # select numeric features only (excludes 'symbol' and 'timestamp')
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # ensure stable order
        self.feature_columns = numeric_cols
        self.n_features = len(self.feature_columns)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size, self.n_features),
                                            dtype=np.float32)

        self.discretization_bins = 21
        self.action_space = spaces.Discrete(self.discretization_bins)
        self._action_center = (self.discretization_bins - 1) // 2

        self.starting_balance = float(starting_balance)
        self.net_worth = self.starting_balance
        self._position = 0.0
        self._step = 0

    def _get_observation(self):
        start_idx = max(0, self._step - self.window_size + 1)
        window = self.df.iloc[start_idx: self._step + 1]
        obs = window[self.feature_columns].to_numpy(dtype=np.float32)
        if obs.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - obs.shape[0], self.n_features), dtype=np.float32)
            obs = np.vstack([pad, obs])

        mean = np.nanmean(obs, axis=0, keepdims=True)
        std = np.nanstd(obs, axis=0, keepdims=True)

        # Epsilon para estabilidade numérica
        epsilon = 1e-8
        obs = np.nan_to_num((obs - mean) / (std + epsilon), nan=0.0, posinf=0.0, neginf=0.0)

        obs = np.clip(obs, -10.0, 10.0)

        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.net_worth = self.starting_balance
        self._position = 0.0
        self._step = 0
        return self._get_observation(), {}

    def step(self, action):
        if self._step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, {}

        prev_price = float(self.df.iloc[self._step]["close"]) if "close" in self.df.columns else 0.0
        self._step += 1
        cur_price = float(self.df.iloc[self._step]["close"]) if "close" in self.df.columns else 0.0

        pos = (int(action) - self._action_center) / float(self._action_center)
        self._position = pos

        if prev_price > 0 and cur_price > 0:
            log_ret = np.log(cur_price / prev_price)
        else:
            log_ret = 0.0

        # Recompensa como log-retorno ponderado pela posição
        reward = self._position * log_ret

        # Clipping da recompensa para evitar valores extremos
        reward = np.clip(reward, -1.0, 1.0) 

        # Atualização correta do net_worth
        self.net_worth = max(0.0, self.net_worth * (1 + reward))

        obs = self._get_observation()
        info = {"net_worth": self.net_worth, "position": self._position, "timestamp": self.df.iloc[self._step].get("timestamp", None)}

        return obs, float(reward), self._step >= len(self.df) - 1, False, info

    def render(self, mode="human"):
        price = float(self.df.iloc[self._step]["close"]) if "close" in self.df.columns else 0.0
        print(f"[Render] Step {self._step} | Price {price:.2f} | Position {self._position:.2f} | NetWorth {self.net_worth:.2f}")

def create_rl_environment(data_frame: pd.DataFrame,
                            pinn_weights_path: str,
                            pinn_scaling_path: str,
                            ticker: str = "TICKER",
                            use_pinn: bool = True) -> TradingEnv:
    """
    Cria o ambiente TradingEnv usando a escala correta do prêmio para o PINN.
    """
    dataset = add_features(data_frame, selic_df=None)

    if use_pinn:
        if not os.path.exists(pinn_weights_path) or not os.path.exists(pinn_scaling_path):
            raise FileNotFoundError(f"Arquivos do PINN não encontrados: {pinn_weights_path} ou {pinn_scaling_path}")

        try:
            scaling = load_scaling_factors(pinn_scaling_path)
            # Carrega a escala do PRÊMIO do arquivo JSON
            P_min = scaling.get("P_min")
            P_max = scaling.get("P_max")
            
            if P_min is None or P_max is None:
                raise ValueError("Fatores de escala do prêmio (P_min, P_max) não encontrados no arquivo de scaling.")

            net = EuropeanCallPINN(S_min=scaling.get("S_min"), S_max=scaling.get("S_max"))
            state = torch.load(pinn_weights_path, map_location=device)
            net.load_state_dict(state)
            net.to(device)
            net.eval()

            feat_cols = ["S_norm", "T_norm", "K_norm", "moneyness_norm", "vol", "r"]
            feats = dataset[feat_cols].fillna(0.0).values.astype("float32")
            
            with torch.no_grad():
                inp = torch.from_numpy(feats).to(device)
                preds_normalized = net(inp).cpu().numpy().reshape(-1)

            # **CORREÇÃO APLICADA AQUI**
            # Desnormaliza a predição usando a escala CORRETA do prêmio
            dataset["pinn_pred"] = preds_normalized * (P_max - P_min) + P_min

            dataset = add_pinn_derived_features(dataset)
            logger.info("[Hermes] PINN pré-treinado carregado e predições (com escala correta) adicionadas.")

        except Exception as e:
            logger.error(f"[Hermes] Falha crítica ao processar o PINN: {e}", exc_info=True)
            raise
    else:
        for col in ["pinn_pred", "pinn_price_ratio", "pinn_momentum_5d", "pinn_vol_10d"]:
            dataset[col] = 0.0

    env = TradingEnv(dataset=dataset, window_size=20, starting_balance=10000.0)
    env.ticker = ticker
    return env
