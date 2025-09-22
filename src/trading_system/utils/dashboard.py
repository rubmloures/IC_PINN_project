# src/trading_system/utils/dashboard.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging # Importa o módulo de logging

# Obtém uma instância do logger para este módulo
logger = logging.getLogger(__name__)

class TrainingDashboard:
    """Gera um resumo formatado do progresso do treinamento."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.episode_logs = []
        self.step_logs = []

    def log_step(self, net_worth, position, reward, pinn_pred=None, premium=None):
        self.step_logs.append({
            "net_worth": net_worth,
            "position": position,
            "reward": reward,
            "pinn_pred": pinn_pred,
            "premium": premium
        })

    def log_episode(self, episode_num, model_logger):
        if not self.step_logs:
            return

        df_steps = pd.DataFrame(self.step_logs)
        start_nw = self.step_logs[0]['net_worth']
        final_nw = self.step_logs[-1]['net_worth']
        
        buys = (df_steps['position'] > 0).sum()
        sells = (df_steps['position'] < 0).sum()
        
        pinn_mae = (df_steps['pinn_pred'] - df_steps['premium']).abs().mean()
        latest_metrics = model_logger.name_to_value
        
        log_entry = {
            "episode": episode_num,
            "final_net_worth": final_nw,
            "valorizacao_percent": ((final_nw / start_nw) - 1) * 100 if start_nw > 0 else 0,
            "reward_medio": df_steps['reward'].mean(),
            "buys": buys,
            "sells": sells,
            "pinn_mae": pinn_mae,
            **latest_metrics
        }
        self.episode_logs.append(log_entry)
        self.print_episode_summary(log_entry)
        self.step_logs = []

    def print_episode_summary(self, log_entry):
        """Imprime o resumo do episódio usando o logger."""
        # Cabeçalho
        logger.info("\n" + "="*80)
        logger.info(f" EPISÓDIO {log_entry['episode']:>3} | {self.ticker} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        logger.info("="*80)
        
        # Resultado Financeiro
        logger.info(" RESULTADO FINANCEIRO:")
        logger.info(f"  ├── Patrimônio Final: R$ {log_entry['final_net_worth']:>10.2f}")
        logger.info(f"  └── Valorização     : {log_entry['valorizacao_percent']:>10.2f} %")
        
        # Operações
        logger.info("\n OPERAÇÕES:")
        logger.info(f"  ├── Compras         : {log_entry['buys']:>5}")
        logger.info(f"  ├── Vendas          : {log_entry['sells']:>5}")
        logger.info(f"  └── Reward Médio    : {log_entry['reward_medio']:>10.4f}")

        # Precisão do PINN
        logger.info("\n PRECISÃO DO PINN (MAE):")
        logger.info(f"  └── Erro Médio (R$): {log_entry.get('pinn_mae', np.nan):>10.4f}")

        # Avaliação do Modelo PPO
        logger.info("\n AVALIAÇÃO DO MODELO PPO:")
        logger.info(f"  ├── Loss (Value)    : {log_entry.get('train/value_loss', np.nan):.4e}")
        logger.info(f"  ├── Entropia        : {log_entry.get('train/entropy_loss', np.nan):.4f}")
        logger.info(f"  └── Learning Rate   : {log_entry.get('train/learning_rate', np.nan):.1e}")
        logger.info("="*80 + "\n")

    def finalize_and_plot(self):
        pass