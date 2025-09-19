# tests/test_features.py
import pytest
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from trading_system.data.feature_engineering import add_features, add_pinn_derived_features

@pytest.fixture(scope="module")
def sample_feature_df():
    """Cria um DataFrame de exemplo para testar a engenharia de features."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    data = {
        'date': dates,
        'open': 100 + np.random.randn(50).cumsum(),
        'high': 102 + np.random.randn(50).cumsum(),
        'low': 98 + np.random.randn(50).cumsum(),
        'close': 101 + np.random.randn(50).cumsum(),
        'pinn_pred': 2.5 + np.random.rand(50) * 0.5,
        'volume': np.random.randint(1000, 5000, size=50)
    }
    return pd.DataFrame(data)

def test_add_technical_features(sample_feature_df):
    """Testa a adição de features técnicas (ex: RSI, Bollinger)."""
    df_featured = add_features(sample_feature_df.copy(), selic_df=None)
    
    assert 'rsi' in df_featured.columns
    assert 'bollinger' in df_featured.columns
    assert pd.api.types.is_numeric_dtype(df_featured['rsi'])
    assert pd.api.types.is_numeric_dtype(df_featured['bollinger'])
    
    # --- CORREÇÃO APLICADA AQUI ---
    # O primeiro valor do RSI é preenchido com 0.0 pela função add_features.
    assert df_featured['rsi'].iloc[0] == 0.0

def test_add_pinn_derived_features(sample_feature_df):
    """Testa a adição de features derivadas da predição do PINN."""
    df_featured = add_pinn_derived_features(sample_feature_df.copy())
    
    assert 'pinn_price_ratio' in df_featured.columns
    assert 'pinn_momentum_5d' in df_featured.columns
    assert pd.api.types.is_numeric_dtype(df_featured['pinn_price_ratio'])
    assert not df_featured['pinn_price_ratio'].isnull().all()