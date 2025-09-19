# scripts/train_initial_pinn.py
import argparse
import json
import logging
import os
import sys
import time
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Adiciona o diretório 'src' ao path para encontrar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_system.models.pinn_architecture import EuropeanCallPINN
from src.trading_system.utils.logger_config import setup_logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(csv_path: str):
    """Carrega e pré-processa os dados de opções."""
    logger.info(f"Carregando e processando dados de {csv_path}...")
    df = pd.read_csv(csv_path)
    df['time_to_maturity'] = df['days_to_maturity'] / 252.0
    df['volatility'].fillna(df['volatility'].mean(), inplace=True)
    df['vol'] = df['volatility'] / 100.0
    df['r'] = 13.75 / 100.0
    df.dropna(inplace=True)
    epsilon = 1e-8
    df['moneyness_ratio'] = df['spot_price'] / (df['strike'] + epsilon)
    return df

def save_scaling_factors(df: pd.DataFrame, path: str):
    """Calcula e salva os fatores de escala em um arquivo JSON."""
    factors = {
        'S_min': df['spot_price'].min(), 'S_max': df['spot_price'].max(),
        'K_min': df['strike'].min(), 'K_max': df['strike'].max(),
        'T_max': df['time_to_maturity'].max(),
        'moneyness_min': df['moneyness_ratio'].min(), 'moneyness_max': df['moneyness_ratio'].max(),
        'P_min': df['premium'].min(), 'P_max': df['premium'].max(),
        # --- CORREÇÃO APLICADA AQUI ---
        'r': df['r'].mean() # Salva a taxa de juros média
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(factors, f, indent=4)
    logger.info(f"Fatores de escala salvos em {path}")
    return factors

def main(args):
    setup_logging()
    
    full_df = preprocess_data(args.data_path)
    scaling_factors = save_scaling_factors(full_df, args.scaling_save_path)

    epsilon = 1e-8
    full_df['S_norm'] = (full_df['spot_price'] - scaling_factors['S_min']) / (scaling_factors['S_max'] - scaling_factors['S_min'] + epsilon)
    full_df['T_norm'] = full_df['time_to_maturity'] / (scaling_factors['T_max'] + epsilon)
    full_df['K_norm'] = (full_df['strike'] - scaling_factors['K_min']) / (scaling_factors['K_max'] - scaling_factors['K_min'] + epsilon)
    full_df['moneyness_norm'] = (full_df['moneyness_ratio'] - scaling_factors['moneyness_min']) / (scaling_factors['moneyness_max'] - scaling_factors['moneyness_min'] + epsilon)

    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    feature_cols = ['S_norm', 'T_norm', 'K_norm', 'moneyness_norm', 'vol', 'r']
    X_train = train_df[feature_cols].values.astype('float32')
    y_train = train_df['premium'].values.reshape(-1, 1).astype('float32')
    X_val = val_df[feature_cols].values.astype('float32')
    y_val = val_df['premium'].values.reshape(-1, 1).astype('float32')

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=args.batch_size)

    # --- CORREÇÃO APLICADA AQUI ---
    model = EuropeanCallPINN(
        S_min=scaling_factors['S_min'], S_max=scaling_factors['S_max'],
        K_min=scaling_factors['K_min'], K_max=scaling_factors['K_max'],
        T_max=scaling_factors['T_max'],
        r=scaling_factors['r'] # Passa a taxa de juros para o modelo
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    logger.info(f"Iniciando treinamento inicial por {args.epochs} épocas...")
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                total_val_loss += loss_fn(y_pred, y_batch).item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.weights_save_path)
            logger.info(f"  Epoch {epoch+1} | Val Loss melhorado: {avg_val_loss:.6f}. Modelo salvo.")

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | T: {time.time()-start_time:.2f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    logger.info(f"Treinamento inicial concluído. Melhor Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento Inicial do Modelo PINN")
    parser.add_argument("--data-path", type=str, required=True, help="Caminho para o CSV com todos os dados de opções.")
    parser.add_argument("--weights-save-path", type=str, default="src/trading_system/models/pinn_weights.pt", help="Onde salvar os pesos do modelo.")
    parser.add_argument("--scaling-save-path", type=str, default="models/pinn_scaling_factors.json", help="Onde salvar os fatores de escala.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)