# scripts/train_pinn_placeholder.py

import torch
import os
from trading_system.models.PINN_model import PINN

def create_placeholder_model():
    """Cria e salva um modelo PINN não treinado para servir como placeholder."""
    
    input_dim = 10 # O número de features que o modelo espera
    model = PINN(input_dim=input_dim)
    
    # Define o caminho para salvar o modelo
    model_dir = "models"
    model_path = os.path.join(model_dir, "pinn_model.pkl")
    
    # Garante que o diretório exista
    os.makedirs(model_dir, exist_ok=True)
    
    # Salva o estado inicial (pesos aleatórios) do modelo
    torch.save(model.state_dict(), model_path)
    
    print(f"Modelo PINN placeholder salvo em: {model_path}")

if __name__ == "__main__":
    create_placeholder_model()