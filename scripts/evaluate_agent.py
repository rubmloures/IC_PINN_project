import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentEval")


def evaluate_agent(
    ticker: str = "PETR4.SA",
    logs_dir: str = "rl_logs",
    pinn_eval_dir: str = "pinn_eval_logs",
    output_dir: str = "agent_eval_results",
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Carregar logs do agente ---
    agent_log_path = os.path.join(logs_dir, f"{ticker.replace('.','_')}_training_log.csv")
    if not os.path.exists(agent_log_path):
        raise FileNotFoundError(f"Log do agente não encontrado: {agent_log_path}")

    df_agent = pd.read_csv(agent_log_path)
    logger.info(f"Logs do agente carregados: {df_agent.shape} linhas")

    # --- Carregar predições do PINN ---
    pinn_eval_path = os.path.join(pinn_eval_dir, f"{ticker.replace('.','_')}_pinn_eval.csv")
    if not os.path.exists(pinn_eval_path):
        raise FileNotFoundError(f"Avaliação do PINN não encontrada: {pinn_eval_path}")

    df_pinn = pd.read_csv(pinn_eval_path)
    logger.info(f"Predições do PINN carregadas: {df_pinn.shape} linhas")

    # --- Alinhar timestamps ---
    df_agent["timestamp"] = pd.to_datetime(df_agent["timestamp"])
    df_pinn["timestamp"] = pd.to_datetime(df_pinn["timestamp"])

    # --- Merge dos dados ---
    df_merged = pd.merge_asof(
        df_agent.sort_values("timestamp"),
        df_pinn.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )

    # --- Salvar merge para debug ---
    merged_path = os.path.join(output_dir, f"{ticker.replace('.','_')}_merged_eval.csv")
    df_merged.to_csv(merged_path, index=False)
    logger.info(f"Dados combinados salvos em {merged_path}")

    # --- Plot comparativo ---
    plt.figure(figsize=(14, 7))
    plt.plot(df_merged["timestamp"], df_merged["close"], label="Preço Real", color="blue")
    plt.plot(df_merged["timestamp"], df_merged["pinn_pred"], label="PINN Predição", color="red", alpha=0.7)
    plt.plot(df_merged["timestamp"], df_merged["net_worth"], label="Agente Net Worth", color="green")

    plt.xlabel("Tempo")
    plt.ylabel("Valor / Preço")
    plt.title(f"Comparação - Agente RL vs PINN vs Preço Real ({ticker})")
    plt.legend()
    plt.grid(True)

    fig_path = os.path.join(output_dir, f"{ticker.replace('.','_')}_agent_vs_pinn_vs_real.png")
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"Gráfico salvo em {fig_path}")
    return df_merged


if __name__ == "__main__":
    evaluate_agent(
        ticker="PETR4.SA",
    )
