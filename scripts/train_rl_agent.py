# scripts/train_rl_agent.py
import os
import argparse
import math
import logging
import numpy as np
import pandas as pd
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from trading_system.data.data_fetcher import fetch_historical_data
from trading_system.data.feature_engineering import add_features, get_selic_series
from trading_system.rl_environment.environment import create_rl_environment

# optional import of PINN trainer (if available)
try:
    from trading_system.models.PINN_model import train_and_save_pinn, load_scaling_factors
except Exception:
    train_and_save_pinn = None
    load_scaling_factors = None

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--ticker", type=str, default="PETR4.SA")
parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
parser.add_argument("--summary-format", type=str, default="B4", help="Escolha: B1..B5 (B4 = Livro Contábil)")
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HermesTrain")


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.last_metrics = {}

    def _on_rollout_end(self) -> None:
        try:
            name_vals = getattr(self.model.logger, "name_to_value", {})
            metrics = {}
            keys = ["train/loss", "train/entropy_loss", "train/explained_variance",
                    "train/policy_gradient_loss", "train/value_loss", "time/fps",
                    "train/clip_fraction", "train/approx_kl"]
            for k in keys:
                if k in name_vals:
                    try:
                        metrics[k] = float(name_vals[k])
                    except Exception:
                        metrics[k] = name_vals[k]
            # grad norm attempt
            grad_norm = float("nan")
            try:
                import torch
                total_sq = 0.0
                found = False
                for p in self.model.policy.parameters():
                    if getattr(p, "grad", None) is not None:
                        found = True
                        total_sq += float(torch.norm(p.grad).item() ** 2)
                if found:
                    grad_norm = math.sqrt(total_sq)
            except Exception:
                pass
            metrics["grad_norm"] = grad_norm
            self.last_metrics = metrics
        except Exception as e:
            logger.debug(f"[Callback] coleta métricas falhou: {e}")
            self.last_metrics = {}

    def _on_step(self) -> bool:
        return True


def plot_agent_results(step_logs: pd.DataFrame, episode_summary: pd.DataFrame, pinn_df, ticker: str):
    import matplotlib.pyplot as plt
    out_dir = "training_results"
    os.makedirs(out_dir, exist_ok=True)
    fig1 = None
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        step_logs["net_worth"].astype(float).plot(ax=ax, title=f"{ticker} - Net Worth")
        figpath = os.path.join(out_dir, f"{ticker.replace('.', '_')}_networth.png")
        fig.tight_layout(); fig.savefig(figpath); plt.close(fig)
        fig1 = figpath
    except Exception:
        fig1 = None

    fig2 = None
    try:
        if pinn_df is not None and {"timestamp", "close", "pinn_pred"}.issubset(pinn_df.columns):
            p = pinn_df.copy(); p["timestamp"] = pd.to_datetime(p["timestamp"])
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(p["timestamp"], p["close"], label="real")
            ax.plot(p["timestamp"], p["pinn_pred"], label="pinn")
            ax.legend()
            fig2 = os.path.join(out_dir, f"{ticker.replace('.', '_')}_pinn_vs_real.png")
            fig.tight_layout(); fig.savefig(fig2); plt.close(fig)
    except Exception:
        fig2 = None

    fig3 = None
    try:
        if not episode_summary.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            if "final_net_worth" in episode_summary.columns:
                ax.plot(episode_summary.index, episode_summary["final_net_worth"], label="final_net_worth")
            for col in ["entropy_mean", "value_loss_mean", "explained_variance"]:
                if col in episode_summary.columns:
                    ax.plot(episode_summary.index, episode_summary[col], label=col)
            ax.legend()
            fig3 = os.path.join(out_dir, f"{ticker.replace('.', '_')}_episode_metrics.png")
            fig.tight_layout(); fig.savefig(fig3); plt.close(fig)
    except Exception:
        fig3 = None

    logger.info(f"[Plots] networth:{fig1} pinn:{fig2} episode:{fig3}")
    return fig1, fig2, fig3


def main():
    ticker = args.ticker
    total_episodes = args.episodes
    steps_per_episode = args.steps
    starting_balance = 10000.0

    os.makedirs("rl_logs", exist_ok=True)
    os.makedirs("training_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    logger.info(f"Carregando dados históricos para {ticker}...")
    df = fetch_historical_data(ticker)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'].dt.year >= 2020) & (df['timestamp'].dt.year <= 2023)].copy()
    logger.info(f"Dados filtrados para o período de 2020 a 2023, total de {len(df)} registros.")
    logger.info(f"{len(df)} registros carregados.")

    # SELIC
    start = df["timestamp"].min().strftime("%Y-%m-%d")
    end = df["timestamp"].max().strftime("%Y-%m-%d")
    selic_df = get_selic_series(start, end)

    # add features (bollinger & vix_proxy included)
    df = add_features(df, selic_df)

    # PINN artifacts (weights + scaling)
    pinn_weights_default = os.path.join("src", "trading_system", "core", "pinn_weights.pt")
    pinn_scaling_default = os.path.join("models", "pinn_scaling_factors.json")

    # if missing, try to train automatically (if train_and_save_pinn available)
    if (not os.path.exists(pinn_weights_default)) or (not os.path.exists(pinn_scaling_default)):
        if train_and_save_pinn is not None:
            logger.info("[Hermes] PINN artifacts ausentes -> treinando automaticamente usando o mesmo DataFrame de input...")
            try:
                train_and_save_pinn(df=df, weights_out=pinn_weights_default, scaling_out=pinn_scaling_default, epochs=50, batch_size=4096, lr=1e-4)
            except Exception as e:
                logger.warning(f"[Hermes] Falha ao treinar PINN automaticamente: {e}")
        else:
            logger.warning("[Hermes] train_and_save_pinn não disponível -> ignorando PINN.")

    # create env with use_pinn=True (will fallback if files missing)
    env = create_rl_environment(data_frame=df, pinn_weights_path=pinn_weights_default, pinn_scaling_path=pinn_scaling_default, ticker=ticker, use_pinn=True)
    vec_env = DummyVecEnv([lambda: env])

    logger.info("Ambiente RL criado.")
    logger.info(f"Observation space: {env.observation_space.shape}, Action space: {env.action_space}")

    model = PPO("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                n_steps=512,          
                batch_size=64,
                learning_rate=1e-4,   
                gamma=0.99
            )

    # CSV headers
    step_log_path = os.path.join("rl_logs", f"{ticker.replace('.', '_')}_training_steps.csv")
    summary_log_path = os.path.join("rl_logs", f"{ticker.replace('.', '_')}_training_summary.csv")
    step_cols = [
        "episode", "step", "timestamp", "price", "pinn_pred", "rsi", "macd",
        "bollinger", "volume", "vix_proxy", "net_worth", "reward", "position",
        "entropy", "value_loss", "policy_loss", "grad_norm"
    ]
    with open(step_log_path, "w", encoding="utf-8") as f:
        f.write(",".join(step_cols) + "\n")
    summary_cols = [
        "episode", "final_net_worth", "total_reward", "buy_count", "sell_count",
        "entropy_mean", "value_loss_mean", "explained_variance", "grad_norm_mean"
    ]
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(",".join(summary_cols) + "\n")

    cb = TrainingMetricsCallback()
    all_step_logs = []
    episode_summaries = []

    logger.info("Iniciando loop de episódios.")
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        rewards = []
        buy_count = 0
        sell_count = 0

        bar = trange(steps_per_episode, desc=f"Ep {ep}", leave=False)
        while not done and step < steps_per_episode:
            action_raw, _ = model.predict(obs, deterministic=False)
            # robust cast
            try:
                action_int = int(np.asarray(action_raw).item())
            except Exception:
                arr = np.asarray(action_raw)
                action_int = int(arr.flatten()[0]) if arr.size else int(action_raw)

            obs, reward, done, truncated, info = env.step(action_int)

            idx = min(env._step, len(env.df) - 1)
            row = env.df.iloc[idx]
            row_ts = row.get("timestamp", "")
            price = float(row.get("close", np.nan)) if "close" in env.df.columns else np.nan
            pinn_pred = float(row.get("pinn_pred", np.nan)) if "pinn_pred" in env.df.columns else np.nan
            rsi = float(row.get("rsi", np.nan))
            macd = float(row.get("macd", np.nan))
            boll = float(row.get("bollinger", np.nan))
            vol = float(row.get("volume", np.nan)) if "volume" in env.df.columns else np.nan
            vix = float(row.get("vix_proxy", np.nan))

            last_metrics = cb.last_metrics if isinstance(cb.last_metrics, dict) else {}
            entropy = last_metrics.get("train/entropy_loss", np.nan)
            val_loss = last_metrics.get("train/value_loss", np.nan)
            pol_loss = last_metrics.get("train/policy_gradient_loss", np.nan)
            grad_norm = last_metrics.get("grad_norm", np.nan)

            # append step CSV
            with open(step_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{ep},{step},{row_ts},{price},{pinn_pred},{rsi},{macd},{boll},{vol},{vix},"
                    f"{info.get('net_worth', np.nan)},{reward},{info.get('position', np.nan)},"
                    f"{entropy},{val_loss},{pol_loss},{grad_norm}\n"
                )

            # decision label
            center = env._action_center
            if action_int < center:
                decision = "Venda"; sell_count += 1
            elif action_int > center:
                decision = "Compra"; buy_count += 1
            else:
                decision = "Mantém"

            logger.debug(f"[Hermes step] Ep{ep} St{step} | Act {action_int}({decision}) | Pos {info.get('position', 0):.2f} | NW {info.get('net_worth', 0):.2f} | r={reward:.4f} | price={price:.2f} | PINN={pinn_pred:.6f}")

            all_step_logs.append({
                "episode": ep, "step": step, "timestamp": row_ts, "price": price, "pinn_pred": pinn_pred,
                "rsi": rsi, "macd": macd, "bollinger": boll, "volume": vol, "vix_proxy": vix,
                "net_worth": info.get("net_worth", np.nan), "reward": reward, "position": info.get("position", np.nan),
                "entropy": entropy, "value_loss": val_loss, "policy_loss": pol_loss, "grad_norm": grad_norm
            })

            rewards.append(float(reward))
            step += 1
            bar.update(1)
        bar.close()

        # learn on collected steps
        logger.info(f"[Hermes] Episódio {ep} coletou {step} steps, iniciando model.learn(...)")
        try:
            model.learn(total_timesteps=step, reset_num_timesteps=False, callback=cb)
        except Exception as e:
            logger.warning(f"[Hermes] model.learn falhou: {e}")

        final_metrics = cb.last_metrics if cb.last_metrics else {}
        final_nw = float(all_step_logs[-1]["net_worth"]) if all_step_logs else starting_balance
        total_reward = float(np.sum(rewards)) if rewards else 0.0
        value_loss_mean = final_metrics.get("train/value_loss", np.nan)
        explained_var = final_metrics.get("train/explained_variance", np.nan)
        grad_norm_mean = final_metrics.get("grad_norm", np.nan)

        patrimony_change = final_nw - starting_balance
        pct_change = (patrimony_change / starting_balance) * 100.0
        cash_flow = float(np.sum([r for r in rewards if r > 0])) if rewards else 0.0
        trans_costs = 0.0

        # B4-style summary (Livro Contábil) or fallback
        if args.summary_format.upper() == "B4":
            logger.info("=" * 60)
            logger.info(f"EPISÓDIO {ep} | STEPS {step}")
            logger.info("=" * 60)
            logger.info("RESULTADO FINANCEIRO:")
            logger.info(f"├── Fluxo de Caixa (somas rewards>0): +R$ {cash_flow:.2f}")
            logger.info(f"├── Patrimônio final: R$ {final_nw:.2f} (inicial R$ {starting_balance:.2f})")
            logger.info(f"├── Valorização: {patrimony_change:+.2f} ({pct_change:.2f}%)")
            logger.info(f"└── Custos Transacionais (estimado): -R$ {trans_costs:.2f}")
            logger.info("")
            logger.info("AVALIAÇÃO DAS MÉTRICAS (último passo):")
            last_step = all_step_logs[-1] if all_step_logs else {}
            logger.info(f"├── PINN_Pred (último): {last_step.get('pinn_pred', np.nan):.6f}")
            logger.info(f"├── RSI (último): {last_step.get('rsi', np.nan):.3f}")
            logger.info(f"├── MACD (último): {last_step.get('macd', np.nan):.3f}")
            logger.info(f"├── Bollinger (último): {last_step.get('bollinger', np.nan):.3f}")
            logger.info(f"├── Volume (último): {last_step.get('volume', np.nan):.0f}")
            logger.info(f"└── VIX proxy (último): {last_step.get('vix_proxy', np.nan):.3f}")
            logger.info("")
            logger.info("AVALIAÇÃO DO MODELO PPO:")
            logger.info(f"├── Loss (value_loss): {value_loss_mean}")
            logger.info(f"├── Entropia (train/entropy_loss): {final_metrics.get('train/entropy_loss', np.nan)}")
            logger.info(f"├── Policy grad loss: {final_metrics.get('train/policy_gradient_loss', np.nan)}")
            logger.info(f"├── Explained variance: {explained_var}")
            logger.info(f"├── Grad norm (approx): {grad_norm_mean}")
            logger.info(f"└── Hiperparâmetros: lr={model.learning_rate}, n_steps={model.n_steps}, "
                        f"batch_size={model.batch_size}, gamma={model.gamma}")
            logger.info("")
            logger.info(f"PROGRESSO EPISÓDIO: Compras {buy_count} | Vendas {sell_count} | Reward médio {np.mean(rewards) if rewards else 0:.4f}")
            logger.info("=" * 60)
        else:
            logger.info(f"[Resumo Hermes] Episódio {ep} finalizado | Net Worth {final_nw:.2f} | Reward médio {np.mean(rewards):.4f} | Compras {buy_count} | Vendas {sell_count}")

        # persist summary
        with open(summary_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{final_nw},{total_reward},{buy_count},{sell_count},{final_metrics.get('train/entropy_loss', np.nan)},"
                f"{value_loss_mean},{explained_var},{grad_norm_mean}\n"
            )

        episode_summaries.append({
            "episode": ep, "final_net_worth": final_nw, "total_reward": total_reward,
            "buy_count": buy_count, "sell_count": sell_count,
            "entropy_mean": final_metrics.get("train/entropy_loss", np.nan),
            "value_loss_mean": value_loss_mean,
            "explained_variance": explained_var,
            "grad_norm_mean": grad_norm_mean
        })

    # end episodes
    df_steps = pd.DataFrame(all_step_logs)
    df_summary = pd.DataFrame(episode_summaries).set_index("episode")
    df_steps.to_csv(os.path.join("training_results", f"{ticker.replace('.', '_')}_steps_full.csv"), index=False)
    df_summary.to_csv(os.path.join("training_results", f"{ticker.replace('.', '_')}_summary.csv"))

    # attempt pinn eval plot
    pinn_eval_path = os.path.join("pinn_eval_logs", f"{ticker.replace('.', '_')}_pinn_eval.csv")
    pinn_df = pd.read_csv(pinn_eval_path) if os.path.exists(pinn_eval_path) else None
    plot_agent_results(df_steps, df_summary, pinn_df, ticker)

    # save final model
    model.save(os.path.join("models", f"agent_ppo_{ticker.replace('.', '_')}.zip"))
    logger.info(f"Agente treinado salvo em models/agent_ppo_{ticker.replace('.', '_')}.zip")
    logger.info("Treinamento finalizado.")


if __name__ == "__main__":
    main()
