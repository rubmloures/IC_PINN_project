# scripts/evaluate_agent.py
import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_system.data.data_merger import get_merged_dataset
from src.trading_system.data.feature_engineering import add_features
from src.trading_system.models.pinn_handler import PINNHandler
from src.trading_system.rl_environment.environment import TradingEnv
from src.trading_system.utils.logger_config import setup_logging

logger = logging.getLogger(__name__)

def run_backtest(env, model):
    """Executa o backtest e retorna um DataFrame com os resultados."""
    obs, _ = env.reset()
    done = False
    results = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Como o ambiente não é vetorizado na avaliação, 'info' é um dicionário direto.
        results.append({
            "date": info.get('timestamp'),
            "net_worth": info.get('net_worth'),
            "position": info.get('position')
        })
    return pd.DataFrame(results)

def main(args):
    setup_logging(level="INFO")
    logger.info("Iniciando avaliação do agente treinado...")

    logger.info("Carregando dataset de avaliação (2024)...")
    eval_df = get_merged_dataset(
        ticker=args.ticker,
        options_csv_path=args.options_data_path,
        start_year=2024,
        end_year=2024
    )
    eval_df = add_features(eval_df, selic_df=None)
    if eval_df.empty:
        logger.error("Dataset de avaliação está vazio. Abortando.")
        return
    logger.info(f"Dataset de avaliação carregado com {len(eval_df)} linhas.")

    pinn_handler = PINNHandler(weights_path=args.pinn_weights_path, scaling_path=args.pinn_scaling_path)
    model = PPO.load(args.agent_path)
    logger.info(f"Agente carregado de {args.agent_path}")

    # Usamos retrain_interval > len(df) para desativar o fine-tuning durante a avaliação
    eval_env = TradingEnv(
        full_dataset=eval_df,
        pinn_handler=pinn_handler,
        retrain_interval=len(eval_df) + 1 
    )
    
    logger.info("Executando backtest...")
    backtest_results = run_backtest(eval_env, model)
    backtest_results.set_index('date', inplace=True)
    
    logger.info("Gerando relatório de performance com QuantStats...")
    returns = backtest_results['net_worth'].pct_change().dropna()
    
    os.makedirs("evaluation_results", exist_ok=True)
    report_path = f"evaluation_results/{args.ticker}_performance_report.html"
    qs.reports.html(returns, benchmark=args.ticker, output=report_path, title=f'{args.ticker} Agent Performance')
    logger.info(f"Relatório de performance completo salvo em: {report_path}")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(backtest_results.index, backtest_results['net_worth'], label='Estratégia do Agente', color='royalblue', lw=2)
    
    # Adiciona o benchmark Buy & Hold
    buy_hold_returns = eval_df['close'].pct_change().dropna()
    buy_hold_equity = (1 + buy_hold_returns).cumprod() * 10000
    ax.plot(buy_hold_equity.index, buy_hold_equity.values, label=f'Buy & Hold ({args.ticker})', color='gray', linestyle='--')
    
    ax.set_title(f'Evolução do Patrimônio: Agente vs. Buy & Hold ({args.ticker})', fontsize=16)
    ax.set_ylabel('Patrimônio (R$)')
    ax.legend()
    
    plot_path = f"evaluation_results/{args.ticker}_net_worth_evolution.png"
    plt.savefig(plot_path)
    logger.info(f"Gráfico de evolução do patrimônio salvo em: {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliador de Agente de Trading")
    parser.add_argument("--ticker", type=str, default="PETR4.SA")
    parser.add_argument("--agent-path", type=str, default="models/agent_ppo_PETR4.SA.zip")
    parser.add_argument("--options-data-path", type=str, default="data/Data_Optionprice_hist/PETR4_2020_24.csv")
    parser.add_argument("--pinn-weights-path", type=str, default="src/trading_system/models/pinn_weights.pt")
    parser.add_argument("--pinn-scaling-path", type=str, default="models/pinn_scaling_factors.json")
    args = parser.parse_args()
    main(args)