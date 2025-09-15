# src/trading_system/core/strategy.py
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
from typing import List, Dict
import logging
import torch
from ..models.PINN_model import PINN
import numpy as np
import os

class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class BaseStrategy(ABC):
    """Classe base abstrata para todas as estratégias de trading."""
    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame) -> Signal:
        pass

class PINN_BlackScholes_Strategy(BaseStrategy):
    """
    Estratégia que utiliza o modelo PINN real para prever movimentos de preço.
    """
    def __init__(self, pinn_model_path: str, input_dim: int):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if os.path.exists(pinn_model_path) and os.path.getsize(pinn_model_path) > 0:
                self.model = PINN(input_dim=input_dim)
                self.model.load_state_dict(torch.load(pinn_model_path, map_location=self.device))
                self.model.eval()
                logging.info(f"Modelo PINN carregado com sucesso de '{pinn_model_path}'.")
            else:
                logging.warning(f"Arquivo de modelo PINN não encontrado ou vazio em '{pinn_model_path}'. A estratégia retornará HOLD.")
        except Exception as e:
            logging.error(f"Falha ao carregar o modelo PINN: {e}", exc_info=True)
            self.model = None

    def _prepare_input_tensor(self, market_data: pd.DataFrame) -> torch.Tensor:
        latest_data = market_data.iloc[-1]
        preco_acao_subjacente = latest_data['close']
        volatilidade = market_data['close'].pct_change().rolling(window=10).std().iloc[-1]

        feature_array = np.array([
            preco_acao_subjacente * 1.05,  # preco_exercicio
            0.05,                          # taxa_juros
            volatilidade,
            0.25,                          # tempo_vencimento
            preco_acao_subjacente,
            latest_data['open'],
            latest_data['high'],
            latest_data['low'],
            latest_data['close'],
            latest_data['volume']
        ]).reshape(1, -1)

        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(feature_array, dtype=torch.float32).to(self.device)

    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Gera um sinal de trading com base na previsão do modelo PINN real.
        Retorna um dicionário: {"signal": Signal, "value": float}
        """
        if self.model is None or market_data.empty or len(market_data) < 10:
            return {"signal": Signal.HOLD, "value": 0.0}

        try:
            input_tensor = self._prepare_input_tensor(market_data)
            with torch.no_grad():
                prediction = self.model(input_tensor)

            prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

            predicted_class = torch.argmax(prediction, dim=1).item()

            if predicted_class == 0:
                signal = Signal.BUY
            elif predicted_class == 1:
                signal = Signal.HOLD
            else:
                signal = Signal.SELL

            return {"signal": signal, "value": float(predicted_class)}

        except Exception as e:
            logging.error(f"Erro durante a geração de sinal do PINN: {e}", exc_info=True)
            return {"signal": Signal.HOLD, "value": 0.0}
