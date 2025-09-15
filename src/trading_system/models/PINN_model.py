# src/trading_system/core/PINN_model.py
# -*- coding: utf-8 -*-
import os
import json
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EuropeanCallPINN(torch.nn.Module):
    """
    Classe compatível com sua versão original (mantive a API).
    A arquitetura aqui é deliberadamente simples/robusta.
    """
    def __init__(self, S_min: float = 0.0, S_max: float = 1.0, K_min: float = 0.0, K_max: float = 1.0, T_max: float = 1.0):
        super().__init__()
        # arquitetura inspirada na sua cópia anterior
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        # scaling (salvo/atualizado no treino)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.K_min = float(K_min)
        self.K_max = float(K_max)
        self.T_max = float(T_max)

    def forward(self, x):
        # x is expected [S_norm, T_norm, K_norm, moneyness_norm, vol, r]
        return self.net(x)

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def train_and_save_pinn(df: Optional[pd.DataFrame] = None,
                        weights_out: str = os.path.join("src", "trading_system", "core", "pinn_weights.pt"),
                        scaling_out: str = os.path.join("models", "pinn_scaling_factors.json"),
                        epochs: int = 1000,
                        batch_size: int = 4096,
                        lr: float = 1e-6) -> Tuple[str, str]:
    """
    Treina um PINN simples a partir do DataFrame `df`.
    Se df for None, tenta carregar dados históricos via data_fetcher.
    Gera `weights_out` e `scaling_out`.
    Retorna (weights_out, scaling_out).
    """
    # se df None, tente carregar dataset via fetch_historical_data se disponível
    if df is None:
        try:
            from trading_system.data.data_fetcher import fetch_historical_data
            logger.info("[PINN] Nenhum df fornecido -> tentando buscar histórico via fetch_historical_data()")
            df = fetch_historical_data("PETR4.SA")
        except Exception:
            raise RuntimeError("train_and_save_pinn: df não fornecido e fetch_historical_data indisponível.")

    # Preprocess mínimo (garantir colunas)
    df = df.copy()
    if "spot_price" not in df.columns and "close" in df.columns:
        df["spot_price"] = df["close"]
    if "strike" not in df.columns:
        df["strike"] = df["close"].rolling(30, min_periods=1).mean().fillna(df.get("close", 0.0))
    if "days_to_maturity" not in df.columns:
        df["days_to_maturity"] = 30
    df["time_to_maturity"] = df["days_to_maturity"] / 252.0

    # simple premium target: use existing column 'premium' or 'pinn_pred' if exists, else approximate
    if "premium" in df.columns:
        premium_col = "premium"
    elif "pinn_pred" in df.columns:
        premium_col = "pinn_pred"
    elif "option_price" in df.columns:
        premium_col = "option_price"
    else:
        # fallback: synthetic premium as 2% of spot (weak, but allows training)
        df["premium"] = 0.02 * df["spot_price"].fillna(df.get("close", 0.0))
        premium_col = "premium"

    # volatility
    if "vol" not in df.columns:
        if "volatility" in df.columns:
            df["vol"] = df["volatility"]
        elif "close" in df.columns:
            df["vol"] = np.log(df["close"] / df["close"].shift(1)).rolling(21, min_periods=1).std().fillna(0.0)
        else:
            df["vol"] = 0.0

    # ensure r exists
    if "r" not in df.columns:
        df["r"] = 0.1375
    df["r"] = df["r"].ffill().fillna(0.1375)

    # moneyness
    eps = 1e-8
    df["moneyness_ratio"] = df["spot_price"] / (df["strike"].replace({0: np.nan}) + eps)

    # normalization factors to save
    S_min = float(df["spot_price"].min())
    S_max = float(df["spot_price"].max() if df["spot_price"].max() > S_min else S_min + 1.0)
    K_min = float(df["strike"].min())
    K_max = float(df["strike"].max() if df["strike"].max() > K_min else K_min + 1.0)
    T_max = float(max(1.0, df["time_to_maturity"].max()))

    # add normalized columns used by PINN
    df["S_norm"] = (df["spot_price"] - S_min) / (S_max - S_min + 1e-8)
    df["T_norm"] = (df["time_to_maturity"]) / (T_max + 1e-8)
    df["K_norm"] = (df["strike"] - K_min) / (K_max - K_min + 1e-8)
    df["moneyness_norm"] = (df["moneyness_ratio"] - df["moneyness_ratio"].min()) / (df["moneyness_ratio"].max() - df["moneyness_ratio"].min() + 1e-8)

    # build dataset for training: require those normalized features + vol + r
    cols = ["S_norm", "T_norm", "K_norm", "moneyness_norm", "vol", "r"]
    df_train = df.dropna(subset=cols + [premium_col]).copy()
    if len(df_train) < 10:
        raise ValueError("Dados insuficientes para treinar o PINN após limpeza de NaNs.")

    X = df_train[cols].values.astype("float32")
    y = df_train[premium_col].values.astype("float32").reshape(-1, 1)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)

    # create model
    _ensure_dir(weights_out)
    _ensure_dir(scaling_out)
    model = EuropeanCallPINN(S_min=S_min, S_max=S_max, K_min=K_min, K_max=K_max, T_max=T_max)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    logger.info(f"[PINN TRAIN] iniciando (epochs={epochs}, batches={len(loader)})")
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"[PINN TRAIN] ep {epoch+1}/{epochs} loss={running_loss/len(loader):.6e}")

    # save weights and scaling
    torch.save(model.state_dict(), weights_out)
    scaling = {"S_min": S_min, "S_max": S_max, "K_min": K_min, "K_max": K_max, "T_max": T_max}
    with open(scaling_out, "w", encoding="utf-8") as f:
        json.dump(scaling, f, indent=2)

    logger.info(f"[PINN TRAIN] pesos salvos em {weights_out}")
    logger.info(f"[PINN TRAIN] scaling salvo em {scaling_out}")
    return weights_out, scaling_out


def load_scaling_factors(path: str = os.path.join("models", "pinn_scaling_factors.json")) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"scaling file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
