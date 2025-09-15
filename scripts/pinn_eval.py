import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trading_system.data.data_fetcher import fetch_historical_data
from trading_system.data.feature_engineering import (
    get_selic_series,
    prepare_pinn_features,
    add_features,
)
from trading_system.models.pinn_loader import load_pinn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PINNEval")

def evaluate_pinn(
    ticker: str = "PETR4.SA",
    pinn_model_path: str = "src/trading_system/core/PINN_model.py",
    pinn_weights_path: str = "src/trading_system/core/pinn_weights.pt",
    output_dir: str = "pinn_eval_logs",
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Carregar dados históricos ---
    logger.info(f"Carregando dados históricos para {ticker}...")
    df = fetch_historical_data(ticker)
    if df is None or df.empty:
        raise ValueError("Nenhum dado encontrado para avaliação.")

    # --- SELIC para normalização ---
    start = df["timestamp"].min().strftime("%Y-%m-%d")
    end = df["timestamp"].max().strftime("%Y-%m-%d")
    selic_df = get_selic_series(start, end)

    # --- Preparar features do PINN ---
    pinn_features = prepare_pinn_features(df, selic_df)
    df = pd.concat([df.reset_index(drop=True), pinn_features.reset_index(drop=True)], axis=1)
    df = add_features(df, selic_df)

    # --- Carregar PINN ---
    pinn = load_pinn(pinn_model_path, pinn_weights_path)

    expected_cols = ["S_norm", "T_norm", "K_norm", "moneyness_norm", "vol", "r"]

    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Coluna {col} ausente no DataFrame, preenchendo com 0.0")
            df[col] = 0.0

    features = df[expected_cols].to_numpy(dtype=np.float32, copy=True)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Predição do PINN ---
    logger.info("🔮 Gerando previsões do PINN...")
    df["pinn_pred"] = pinn.predict(features)

    # --- Salvar resultados ---
    results_path = os.path.join(output_dir, f"{ticker.replace('.','_')}_pinn_eval.csv")
    df[["timestamp", "close", "pinn_pred"]].to_csv(results_path, index=False)
    logger.info(f"Resultados salvos em {results_path}")

    # --- Plotar comparativo ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Preço Real", color="blue")
    plt.plot(df["timestamp"], df["pinn_pred"], label="PINN Predição", color="red", alpha=0.7)
    plt.xlabel("Tempo")
    plt.ylabel("Preço")
    plt.title(f"Comparação PINN vs Preço Real - {ticker}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{ticker.replace('.','_')}_pinn_vs_real.png"))
    plt.close()

    logger.info(f"Gráfico salvo em {output_dir}")
    return df


if __name__ == "__main__":
    evaluate_pinn(
        ticker="PETR4.SA",
    )
