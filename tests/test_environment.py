# tests/test_environment.py
import pytest
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from trading_system.rl_environment.environment import TradingEnv
from trading_system.models.pinn_handler import PINNHandler

@pytest.fixture(scope="module")
def sample_dataset():
    """Cria um dataset mais longo para testar o ambiente."""
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        'date': dates,
        'close': 100 + np.random.randn(100).cumsum(),
        'strike': 100,
        'days_to_maturity': np.arange(100, 0, -1),
        'volatility': 20 + np.random.randn(100) * 2,
        'r': 0.1375,
        'premium': 2.0 + np.random.rand(100)
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def pinn_handler_for_env():
    """Cria uma instância do PINNHandler para ser usada pelo ambiente."""
    weights_path = "src/trading_system/models/pinn_weights.pt"
    scaling_path = "models/pinn_scaling_factors.json"
    if not os.path.exists(weights_path) or not os.path.exists(scaling_path):
        pytest.fail("Arquivos do PINN necessários para o teste de ambiente não encontrados.")
    return PINNHandler(weights_path=weights_path, scaling_path=scaling_path)

@pytest.fixture
def trading_env(sample_dataset, pinn_handler_for_env):
    """Cria uma instância do TradingEnv para cada teste."""
    return TradingEnv(
        full_dataset=sample_dataset,
        pinn_handler=pinn_handler_for_env,
        retrain_interval=30, # Intervalo de fine-tuning para teste
        window_size=10
    )

def test_env_initialization(trading_env):
    """Testa se o ambiente é inicializado corretamente."""
    assert trading_env is not None
    assert trading_env.df is not None
    assert 'pinn_pred' in trading_env.df.columns, "Coluna 'pinn_pred' não foi adicionada ao DataFrame do ambiente."
    assert trading_env.action_space.n == 21
    assert trading_env.observation_space.shape == (10, trading_env.n_features)

def test_env_reset(trading_env):
    """Testa o método reset do ambiente."""
    obs, info = trading_env.reset()
    assert isinstance(obs, np.ndarray), "A observação retornada pelo reset não é um array numpy."
    assert obs.shape == trading_env.observation_space.shape
    assert info['net_worth'] == trading_env.starting_balance, "O patrimônio não foi resetado corretamente."
    assert trading_env._current_step == 0

def test_env_step(trading_env):
    """Testa um único passo no ambiente."""
    trading_env.reset()
    action = trading_env.action_space.sample()
    obs, reward, terminated, truncated, info = trading_env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == trading_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert 'net_worth' in info
    assert trading_env._current_step == 1