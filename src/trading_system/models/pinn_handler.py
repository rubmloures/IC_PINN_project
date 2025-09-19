# src/trading_system/models/pinn_handler.py

import json
import logging
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Importa a arquitetura do novo módulo
from .pinn_architecture import EuropeanCallPINN

# Configuração do Logger
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINNHandler:
    """
    Gerencia todo o ciclo de vida do modelo PINN, incluindo carregamento,
    predição e fine-tuning dinâmico.
    """

    def __init__(self, weights_path: str, scaling_path: str):
        """
        Inicializa o handler, carrega os fatores de escala e o modelo pré-treinado.

        Args:
            weights_path (str): Caminho para o arquivo de pesos do modelo (.pt).
            scaling_path (str): Caminho para o arquivo JSON com os fatores de escala.
        """
        logger.info("Inicializando o PINNHandler...")
        if not os.path.exists(weights_path) or not os.path.exists(scaling_path):
            raise FileNotFoundError(f"Arquivos do PINN não encontrados: {weights_path} ou {scaling_path}")

        self.weights_path = weights_path
        self.scaling_path = scaling_path
        self.scaling_factors = self._load_scaling_factors()

        # Instancia a arquitetura do modelo
        self.model = EuropeanCallPINN(
            S_min=self.scaling_factors.get("S_min", 0),
            S_max=self.scaling_factors.get("S_max", 1),
            K_min=self.scaling_factors.get("K_min", 0),
            K_max=self.scaling_factors.get("K_max", 1),
            T_max=self.scaling_factors.get("T_max", 1.0),
            r=self.scaling_factors.get("r", 0.1375) # Assume uma 'r' média se não estiver nos fatores
        )

        # Carrega os pesos pré-treinados
        self.load_weights(self.weights_path)
        self.model.to(device)
        self.model.eval()
        logger.info(f"Modelo PINN carregado de '{weights_path}' e pronto para predição.")

    def _load_scaling_factors(self) -> Dict[str, Any]:
        """Carrega os fatores de escala do arquivo JSON."""
        with open(self.scaling_path, 'r') as f:
            factors = json.load(f)
        logger.info(f"Fatores de escala carregados de '{self.scaling_path}'.")
        return factors

    def _save_scaling_factors(self):
        """Salva os fatores de escala atualizados no arquivo JSON."""
        with open(self.scaling_path, 'w') as f:
            json.dump(self.scaling_factors, f, indent=4)
        logger.info(f"Fatores de escala atualizados e salvos em '{self.scaling_path}'.")

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza as colunas necessárias do DataFrame para a entrada do modelo.
        Garante que o DataFrame de entrada tenha as colunas esperadas.
        """
        df_norm = df.copy()
        
        column_mapping = {'close': 'spot_price'}
        df_norm.rename(columns=column_mapping, inplace=True)

        epsilon = 1e-8
        df_norm['time_to_maturity'] = df_norm['days_to_maturity'] / 252.0
        df_norm['moneyness_ratio'] = df_norm['spot_price'] / (df_norm['strike'] + epsilon)
        df_norm['vol'] = df_norm.get('volatility', 0) / 100.0
        df_norm['r'] = df_norm.get('r', 0.1375)
        
        df_norm['S_norm'] = (df_norm['spot_price'] - self.scaling_factors['S_min']) / (self.scaling_factors['S_max'] - self.scaling_factors['S_min'] + epsilon)
        df_norm['T_norm'] = df_norm['time_to_maturity'] / (self.scaling_factors['T_max'] + epsilon)
        df_norm['K_norm'] = (df_norm['strike'] - self.scaling_factors['K_min']) / (self.scaling_factors['K_max'] - self.scaling_factors['K_min'] + epsilon)
        df_norm['moneyness_norm'] = (df_norm['moneyness_ratio'] - self.scaling_factors['moneyness_min']) / (self.scaling_factors['moneyness_max'] - self.scaling_factors['moneyness_min'] + epsilon)

        return df_norm
        
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Realiza a predição do prêmio da opção para um conjunto de dados.
        """
        if data_df.empty:
            return np.array([])
            
        self.model.eval()
        df_norm = self._normalize_data(data_df)
        
        feature_cols = ['S_norm', 'T_norm', 'K_norm', 'moneyness_norm', 'vol', 'r']
        features = df_norm[feature_cols].values.astype(np.float32)
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(features).to(device)
            predictions_raw = self.model(input_tensor).cpu().numpy().flatten()
            
        return predictions_raw

    # --- CORREÇÃO APLICADA AQUI ---
    def fine_tune(self, recent_data: pd.DataFrame, epochs: int = 5, learning_rate: float = 1e-4):
        """
        Realiza o fine-tuning do modelo com dados recentes.
        """
        logger.info(f"Iniciando fine-tuning do PINN com {len(recent_data)} amostras por {epochs} épocas...")
        self.model.train()

        df_norm = self._normalize_data(recent_data)
        
        feature_cols = ['S_norm', 'T_norm', 'K_norm', 'moneyness_norm', 'vol', 'r']
        X_train = df_norm[feature_cols].values.astype(np.float32)
        y_train = df_norm['premium'].values.reshape(-1, 1).astype(np.float32)

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0:
                 logger.debug(f"  Fine-tune Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

        self.model.eval()
        self.save_weights(self.weights_path)
        logger.info(f"Fine-tuning concluído. Perda final: {avg_loss:.6f}. Pesos atualizados salvos.")

        self._update_scaling_factors(recent_data)
        self._save_scaling_factors()

    def _update_scaling_factors(self, new_data: pd.DataFrame):
        """
        Recalcula os fatores de escala com base nos dados históricos + novos dados.
        """
        logger.info("Recalculando fatores de escala com novos dados...")
        epsilon = 1e-8
        
        new_data_mapped = new_data.rename(columns={'close': 'spot_price'})

        self.scaling_factors['S_min'] = min(self.scaling_factors['S_min'], new_data_mapped['spot_price'].min())
        self.scaling_factors['S_max'] = max(self.scaling_factors['S_max'], new_data_mapped['spot_price'].max())
        self.scaling_factors['K_min'] = min(self.scaling_factors['K_min'], new_data_mapped['strike'].min())
        self.scaling_factors['K_max'] = max(self.scaling_factors['K_max'], new_data_mapped['strike'].max())
        
        new_time_to_maturity = new_data_mapped['days_to_maturity'] / 252.0
        self.scaling_factors['T_max'] = max(self.scaling_factors['T_max'], new_time_to_maturity.max())
        
        new_moneyness = new_data_mapped['spot_price'] / (new_data_mapped['strike'] + epsilon)
        self.scaling_factors['moneyness_min'] = min(self.scaling_factors['moneyness_min'], new_moneyness.min())
        self.scaling_factors['moneyness_max'] = max(self.scaling_factors['moneyness_max'], new_moneyness.max())

        self.scaling_factors['P_min'] = min(self.scaling_factors.get('P_min', float('inf')), new_data_mapped['premium'].min())
        self.scaling_factors['P_max'] = max(self.scaling_factors.get('P_max', float('-inf')), new_data_mapped['premium'].max())

    def load_weights(self, path: str):
        """Carrega os pesos do modelo de um arquivo."""
        try:
            self.model.load_state_dict(torch.load(path, map_location=device))
            logger.info(f"Pesos do modelo carregados de '{path}'.")
        except Exception as e:
            logger.error(f"Falha ao carregar os pesos de '{path}': {e}", exc_info=True)
            raise

    def save_weights(self, path: str):
        """Salva os pesos atuais do modelo em um arquivo."""
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Pesos do modelo salvos em '{path}'.")
        except Exception as e:
            logger.error(f"Falha ao salvar os pesos em '{path}': {e}", exc_info=True)