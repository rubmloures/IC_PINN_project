# tests/test_environment.py
import logging
import os
import sys
import pandas as pd

# Adiciona o diretório src ao path para permitir importações diretas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from trading_system.data.data_fetcher import fetch_historical_data
from trading_system.rl_environment.environment import create_rl_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ticker = "PETR4.SA"
    logger.info(f"Testando ambiente Hermes com {ticker}...")

    # 1. Carregar e Filtrar Dados
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

    # 2. Caminhos do PINN
    pinn_weights_path = os.path.join("src", "trading_system", "core", "pinn_weights.pt")
    pinn_scaling_path = os.path.join("models", "pinn_scaling_factors.json")

    # 3. Criar ambiente
    env = create_rl_environment(
        data_frame=df,
        pinn_weights_path=pinn_weights_path,
        pinn_scaling_path=pinn_scaling_path,
        ticker=ticker,
    )

    # 4. Teste de reset
    obs, _ = env.reset()
    logger.info(f"Obs inicial (shape {obs.shape}):\n{obs[-1]}")

    # 5. Executar alguns steps aleatórios
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        logger.info(
            f"Step {i+1} | Action {action} | Reward {reward:.4f} | "
            f"Net Worth {info['net_worth']:.2f} | Pos {info['position']:.2f}"
        )
        if done:
            logger.info("Ambiente terminou o episódio.")
            break

    env.render()


if __name__ == "__main__":
    main()