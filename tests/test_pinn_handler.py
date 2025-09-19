# tests/test_pinn_handler.py
import pytest
import os
import sys
import pandas as pd
import numpy as np
import torch

# Adiciona o diretório 'src' ao path para encontrar os módulos do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from trading_system.models.pinn_handler import PINNHandler

# Usa o pytest fixture para criar e destruir objetos de teste de forma limpa
@pytest.fixture(scope="module")
def pinn_handler():
    """Fixture para criar uma instância do PINNHandler para os testes."""
    weights_path = "src/trading_system/models/pinn_weights.pt"
    scaling_path = "models/pinn_scaling_factors.json"

    if not os.path.exists(weights_path) or not os.path.exists(scaling_path):
        pytest.fail(f"Arquivos necessários para o teste não encontrados: {weights_path} ou {scaling_path}")

    return PINNHandler(weights_path=weights_path, scaling_path=scaling_path)

@pytest.fixture(scope="module")
def sample_data():
    """Fixture para criar um DataFrame de exemplo com dados realistas."""
    data = {
        'date': pd.to_datetime(['2023-01-02', '2023-01-03']),
        'close': [100.0, 102.0],
        'strike': [100.0, 105.0],
        'days_to_maturity': [30, 29],
        'volatility': [20.0, 21.0],
        'r': [0.1375, 0.1375],
        'premium': [2.5, 3.0]
    }
    return pd.DataFrame(data)

def test_pinn_handler_initialization(pinn_handler):
    """Testa se o PINNHandler é inicializado corretamente."""
    assert pinn_handler is not None
    assert pinn_handler.model is not None
    assert isinstance(pinn_handler.scaling_factors, dict)
    assert 'S_max' in pinn_handler.scaling_factors

def test_predict(pinn_handler, sample_data):
    """Testa a função de predição."""
    predictions = pinn_handler.predict(sample_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(sample_data)
    assert not np.isnan(predictions).any()

def test_fine_tune(pinn_handler, sample_data):
    """Testa a função de fine-tuning."""
    initial_state_dict = {k: v.clone() for k, v in pinn_handler.model.state_dict().items()}

    # --- CORREÇÃO APLICADA AQUI ---
    # Usamos uma learning_rate alta para garantir uma mudança detectável nos pesos.
    pinn_handler.fine_tune(sample_data, epochs=2, learning_rate=0.01)

    tuned_state_dict = pinn_handler.model.state_dict()

    weights_changed = False
    for key in initial_state_dict:
        if not torch.equal(initial_state_dict[key], tuned_state_dict[key]):
            weights_changed = True
            break

    assert weights_changed, "Os pesos do modelo não foram alterados após o fine-tuning."