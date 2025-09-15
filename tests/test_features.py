# scripts/test_features.py
import logging
import os
import pandas as pd
import numpy as np
from trading_system.data.data_fetcher import fetch_historical_data
from trading_system.data.feature_engineering import add_features, add_pinn_derived_features
from trading_system.models.pinn_loader import load_pinn
from trading_system.models.PINN_model import load_scaling_factors 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    ticker = "PETR4.SA"
    logger.info(f" Testando feature engineering com {ticker}...")

    # --- Bloco 1: Carregar e Filtrar Dados ---
    df = fetch_historical_data(ticker)
    if df.empty:
        logger.error("Nenhum dado foi carregado. Abortando o teste.")
        return

    # Adicionado filtro de data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'].dt.year >= 2020) & (df['timestamp'].dt.year <= 2023)].copy()
    logger.info(f"Dados filtrados para o período de 2020 a 2023, total de {len(df)} registros.")

    if df.empty:
        logger.warning("O DataFrame ficou vazio após a filtragem por data. Verifique o período.")
        return

    # --- Bloco 2: Adicionar Features ---
    df = add_features(df, selic_df=None)

    # --- Bloco 3: Carregar PINN e Gerar Predições ---
    pinn_weights_path = os.path.join("src", "trading_system", "core", "pinn_weights.pt")
    pinn_scaling_path = os.path.join("models", "pinn_scaling_factors.json")
    
    # Usando o pinn_loader para consistência
    pinn_model_path = os.path.join("src", "trading_system", "models", "PINN_model.py")


    if not os.path.exists(pinn_weights_path) or not os.path.exists(pinn_scaling_path):
        logger.error("Arquivos de pesos ou scaling do PINN não encontrados. Abortando.")
        return

    try:
        pinn = load_pinn(pinn_model_path, pinn_weights_path)
        scaling_factors = load_scaling_factors(path=pinn_scaling_path)
        S_max = scaling_factors.get("S_max", 1.0)
        S_min = scaling_factors.get("S_min", 0.0)

        features = df[["S_norm", "T_norm", "K_norm", "moneyness_norm", "vol", "r"]].to_numpy(dtype=np.float32)
        preds_normalized = pinn.predict(features)
        
        df["pinn_pred"] = preds_normalized * (S_max - S_min) + S_min
        logger.info("Previsões do PINN geradas e de-normalizadas.")
        
        # Adicionar features derivadas do PINN
        df = add_pinn_derived_features(df)
        
        logger.info("Primeiras linhas do DataFrame final com todas as features:")
        print(df[["timestamp", "close", "pinn_pred", "pinn_price_ratio", "pinn_momentum_5d"]].head(10))

    except Exception as e:
        logger.error(f"Falha ao processar o PINN: {e}", exc_info=True)


if __name__ == "__main__":
    main()