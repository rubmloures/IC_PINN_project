# src/trading_system/data/feature_engineering.py
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

LOCAL_SELIC_PATH = r"D:\UERJ\Programacao_e_Codigos\trading_system\data\taxa_selic.csv"
BCB_API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"

# --- Indicador Taxa SELIC ---
def get_selic_series(start: str, end: str) -> pd.DataFrame:
    """
    Busca taxa SELIC: CSV local -> BCB API -> fallback fixo (13.75%).
    Retorna DataFrame com colunas ['data','r'] (r em decimal, e.g. 0.1375).
    """
    # 1) CSV local
    try:
        if os.path.exists(LOCAL_SELIC_PATH):
            df_local = pd.read_csv(LOCAL_SELIC_PATH)
            if {"data", "valor"}.issubset(df_local.columns):
                df_local["data"] = pd.to_datetime(df_local["data"], format="%d/%m/%Y", errors="coerce")
                df_local["r"] = pd.to_numeric(df_local["valor"], errors="coerce") / 100.0
                df_local = df_local.dropna(subset=["data", "r"]).sort_values("data").drop_duplicates("data")
                logger.info(f"[Hermes] SELIC carregada do CSV local: {LOCAL_SELIC_PATH}")
                return df_local[["data", "r"]]
            else:
                logger.warning("Falha ao carregar SELIC local: 'data' ou 'valor' ausente no CSV.")
    except Exception as e:
        logger.warning(f"Falha ao carregar SELIC local: {e}")

    # 2) API do BCB
    try:
        base_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        dfs = []
        cursor = start_dt
        while cursor <= end_dt:
            chunk_end = min(cursor + timedelta(days=365 * 5), end_dt)
            url = f"{base_url}?formato=json&dataInicial={cursor.strftime('%d/%m/%Y')}&dataFinal={chunk_end.strftime('%d/%m/%Y')}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data_json = resp.json()
            if data_json:
                chunk_df = pd.DataFrame(data_json)
                chunk_df["data"] = pd.to_datetime(chunk_df["data"], format="%d/%m/%Y", errors="coerce")
                chunk_df["r"] = pd.to_numeric(chunk_df["valor"], errors="coerce") / 100.0
                dfs.append(chunk_df[["data", "r"]])
            cursor = chunk_end + timedelta(days=1)

        if dfs:
            df_api = pd.concat(dfs).dropna().drop_duplicates("data").sort_values("data")
            logger.info("[Hermes] SELIC carregada da API do BCB.")
            return df_api
    except Exception as e:
        logger.error(f"Erro ao buscar SELIC: {e}")

    # 3) fallback
    logger.warning("Usando fallback da SELIC em 13.75% ao ano.")
    idx = pd.date_range(start, end, freq="B")
    return pd.DataFrame({"data": idx, "r": 0.1375})

# --- indicadores auxiliares ---
def compute_volatility(prices: pd.Series, window: int = 21) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window, min_periods=1).std() * np.sqrt(252.0)
    return vol.fillna(method="bfill").fillna(0.0)

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.Series:
    lowest_low = low.rolling(window=k, min_periods=1).min()
    highest_high = high.rolling(window=k, min_periods=1).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    return k_percent.rolling(window=d, min_periods=1).mean().fillna(0.0)

def compute_momentum(series: pd.Series, window: int = 10) -> pd.Series:
    return series.diff(window).fillna(0.0)

# --- NOVOS INDICADORES ---
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean().fillna(method="bfill").fillna(0.0)

def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calcula o Chaikin Money Flow (CMF)."""
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
    cmf = mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return cmf.fillna(0.0)

# --- PINN features preparation (keeps same column names used elsewhere) ---
def prepare_pinn_features(df: pd.DataFrame, selic_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    if selic_df is not None and "data" in selic_df.columns:
        selic_df = selic_df.copy()
        selic_df["data"] = pd.to_datetime(selic_df["data"]).dt.tz_localize(None)
        df = df.merge(selic_df, how="left", left_on="timestamp", right_on="data")
        df.drop(columns=["data"], inplace=True, errors="ignore")

    if "r" not in df.columns:
        df["r"] = 0.1375
    df["r"] = df["r"].ffill().fillna(0.1375)

    if "spot_price" not in df.columns and "close" in df.columns:
        df["spot_price"] = df["close"]
    if "strike" not in df.columns:
        df["strike"] = df["close"].rolling(30, min_periods=1).mean().fillna(df.get("close", 0.0))

    df["days_to_maturity"] = df.get("days_to_maturity", 30)
    df["time_to_maturity"] = df["days_to_maturity"] / 252.0

    if "vol" not in df.columns:
        if "volatility" in df.columns:
            df["vol"] = df["volatility"]
        elif "close" in df.columns:
            df["vol"] = compute_volatility(df["close"])
        else:
            df["vol"] = 0.0

    eps = 1e-8
    df["moneyness_ratio"] = df["spot_price"] / (df["strike"].replace({0: np.nan}) + eps)

    df["S_norm"] = (df["spot_price"] - df["spot_price"].min()) / (df["spot_price"].max() - df["spot_price"].min() + 1e-8)
    df["T_norm"] = (df["time_to_maturity"]) / (df["time_to_maturity"].max() + 1e-8)
    df["K_norm"] = (df["strike"] - df["strike"].min()) / (df["strike"].max() - df["strike"].min() + 1e-8)
    df["moneyness_norm"] = (df["moneyness_ratio"] - df["moneyness_ratio"].min()) / (df["moneyness_ratio"].max() - df["moneyness_ratio"].min() + 1e-8)

    out = df[["S_norm", "T_norm", "K_norm", "moneyness_norm", "vol", "r"]].copy()
    out = out.fillna(0.0)
    return out

# --- After we have pinn_pred column, derive sentiment features ---
def add_pinn_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pinn_pred" not in df.columns:
        df["pinn_pred"] = 0.0
    df["pinn_price_ratio"] = df["pinn_pred"] / (df.get("close", df.get("spot_price", 1.0)) + 1e-8)
    df["pinn_momentum_5d"] = df["pinn_pred"].pct_change(5).fillna(0.0)
    df["pinn_vol_10d"] = df["pinn_pred"].rolling(10, min_periods=1).std().fillna(0.0)
    return df

# --- Main add_features that augment the input DF with many indicators ---
def add_features(df: pd.DataFrame, selic_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    try:
        out = df.copy()

        if 'timestamp' in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"])

        pinn_feats = prepare_pinn_features(out, selic_df)
        for c in pinn_feats.columns:
            out[c] = pinn_feats[c].values

        if "close" in out.columns:
            out["EMA_21"] = out["close"].ewm(span=21, adjust=False).mean()
            out["EMA_50"] = out["close"].ewm(span=50, adjust=False).mean()
            out["rsi"] = compute_rsi(out["close"])
            out["macd"] = compute_macd(out["close"])
            out["momentum"] = compute_momentum(out["close"])
            out["volatility"] = compute_volatility(out["close"])
        else:
            logger.warning("close not in df -> skipping RSI/MACD/volatility")

        if {"high", "low", "close"}.issubset(out.columns):
            out["stochastic"] = compute_stochastic(out["high"], out["low"], out["close"])
            # ADICIONADO ATR
            out["atr"] = compute_atr(out["high"], out["low"], out["close"])
        else:
            out["stochastic"] = 0.0
            out["atr"] = 0.0

        if {"close", "volume", "high", "low"}.issubset(out.columns):
            out["obv"] = compute_obv(out["close"], out["volume"])
            # ADICIONADO CMF
            out["cmf"] = compute_cmf(out["high"], out["low"], out["close"], out["volume"])
        else:
            out["obv"] = 0.0
            out["cmf"] = 0.0

        try:
            ma20 = out["close"].rolling(20, min_periods=1).mean()
            std20 = out["close"].rolling(20, min_periods=1).std().fillna(0.0)
            out["bollinger"] = (out["close"] - ma20) / (std20 + 1e-8)
        except Exception:
            out["bollinger"] = 0.0

        try:
            out["vix_proxy"] = out["volatility"].fillna(0.0) * 100.0
        except Exception:
            out["vix_proxy"] = 0.0

        out = out.fillna(0.0)
        logger.info("[Hermes] Features financeiras + PINN (preparadas) adicionadas com sucesso.")
        return out

    except Exception as e:
        logger.error(f"Erro ao adicionar features: {e}", exc_info=True)
        return df.fillna(0.0)