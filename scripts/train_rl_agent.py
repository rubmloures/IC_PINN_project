# scripts/train_rl_agent.py
import argparse
import logging
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_system.data.data_merger import get_merged_dataset
from src.trading_system.data.feature_engineering import add_features
from src.trading_system.models.pinn_handler import PINNHandler
from src.trading_system.rl_environment.environment import TradingEnv
from src.trading_system.utils.dashboard import TrainingDashboard
from src.trading_system.utils.logger_config import setup_logging

logger = logging.getLogger(__name__)

class DashboardCallback(BaseCallback):
    def __init__(self, dashboard: TrainingDashboard, verbose: int = 0):
        super().__init__(verbose)
        self.dashboard = dashboard

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if info:
            self.dashboard.log_step(
                net_worth=info.get("net_worth"),
                position=info.get("position"),
                reward=self.locals['rewards'][0],
                pinn_pred=info.get("pinn_pred"),
                premium=info.get("premium")
            )
        return True

def main(args):
    setup_logging(level=args.log_level)
    logger.info("Iniciando o processo de treinamento do agente RL...")
    os.makedirs(args.model_save_path, exist_ok=True)

    logger.info("Carregando e unindo os datasets...")
    full_df = get_merged_dataset(
        ticker=args.ticker,
        options_csv_path=args.options_data_path,
        start_year=2020,
        end_year=2023
    )
    full_df = add_features(full_df, selic_df=None)
    logger.info(f"Dataset final criado com {len(full_df)} linhas.")

    logger.info("Inicializando o PINNHandler...")
    pinn_handler = PINNHandler(
        weights_path=args.pinn_weights_path,
        scaling_path=args.pinn_scaling_path
    )

    logger.info("Criando o ambiente de RL com o PINNHandler dinâmico...")
    env = TradingEnv(
        full_dataset=full_df,
        pinn_handler=pinn_handler,
        retrain_interval=args.retrain_interval
    )
    env = DummyVecEnv([lambda: env])

    logger.info("Configurando o agente PPO...")
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=args.log_path,
                n_steps=512, batch_size=64, learning_rate=args.learning_rate,
                gamma=0.99, device="cuda" if args.use_gpu else "cpu")

    dashboard = TrainingDashboard(ticker=args.ticker)
    callback = DashboardCallback(dashboard)
    
    # O número de passos em um episódio é o tamanho do dataset - 1
    total_timesteps_per_episode = len(full_df) - 1
    logger.info(f"Iniciando treinamento por {args.episodes} episódios ({total_timesteps_per_episode} passos por episódio)...")

    for ep in range(1, args.episodes + 1):
        model.learn(
            total_timesteps=total_timesteps_per_episode,
            callback=callback,
            reset_num_timesteps=(ep == 1)
        )
        dashboard.log_episode(ep, model.logger)

    final_model_path = os.path.join(args.model_save_path, f"agent_ppo_{args.ticker}.zip")
    model.save(final_model_path)
    logger.info(f"Treinamento concluído! Agente salvo em: {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinador do Agente de Trading com PINN Dinâmico")
    
    parser.add_argument("--ticker", type=str, default="PETR4.SA")
    parser.add_argument("--options-data-path", type=str, default="data/Data_Optionprice_hist/PETR4_2020_24.csv")
    parser.add_argument("--pinn-weights-path", type=str, default="src/trading_system/models/pinn_weights.pt")
    parser.add_argument("--pinn-scaling-path", type=str, default="models/pinn_scaling_factors.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--retrain-interval", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Taxa de aprendizado para o agente PPO.")
    parser.add_argument("--model-save-path", type=str, default="models/")
    parser.add_argument("--log-path", type=str, default="tensorboard_logs/")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)