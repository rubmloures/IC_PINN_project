# src/trading_system/data/data_fetcher.py

import pandas as pd
import logging
from trading_system.db.database_manager import db_manager
import numpy as np

logger = logging.getLogger(__name__)

def normalize_symbol(symbol: str) -> str:
    """
    Normaliza o símbolo para o formato usado no banco.
    Exemplo: PETR4.SA -> PETR4
    """
    return symbol.replace(".SA", "").upper().strip()

def fetch_historical_data(symbol: str) -> pd.DataFrame:
    """
    Busca os dados históricos do ativo no banco.
    """
    normalized_symbol = normalize_symbol(symbol)
    query = """
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM historical_market_data
        WHERE symbol = %s
        ORDER BY timestamp ASC;
    """
    try:
        with db_manager.get_connection() as conn:
            df = pd.read_sql(query, conn, params=(normalized_symbol,))
        if df.empty:
            logger.warning(f"Nenhum dado encontrado para o símbolo '{symbol}' (normalizado para '{normalized_symbol}').")
        else:
            logger.info(f"Carregados {len(df)} registros para o símbolo '{symbol}' (normalizado para '{normalized_symbol}').")
        return df
    except Exception as e:
        logger.error(f"Erro ao buscar dados históricos para {symbol}: {e}", exc_info=True)
        return pd.DataFrame()
