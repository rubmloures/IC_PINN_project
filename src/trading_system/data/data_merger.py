# src/trading_system/data/data_merger.py
import pandas as pd
import logging
from .data_fetcher import fetch_historical_data

logger = logging.getLogger(__name__)

def get_merged_dataset(ticker: str, options_csv_path: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Busca os dados de mercado do banco, carrega os dados de opções de um CSV e
    os une, selecionando a opção mais relevante para cada dia de pregão.

    Args:
        ticker (str): O ticker do ativo (ex: "PETR4.SA").
        options_csv_path (str): O caminho para o arquivo CSV com os dados históricos de opções.
        start_year (int): Ano de início do filtro.
        end_year (int): Ano de fim do filtro.

    Returns:
        pd.DataFrame: Um DataFrame unificado contendo dados de mercado e de opções.
    """
    logger.info("Iniciando a criação do dataset unificado...")

    # 1. Buscar e filtrar os dados de mercado (COTAHIST do banco)
    market_df = fetch_historical_data(ticker)
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp']).dt.date
    market_df = market_df[
        (pd.to_datetime(market_df['timestamp']).dt.year >= start_year) &
        (pd.to_datetime(market_df['timestamp']).dt.year <= end_year)
    ].copy()
    logger.info(f"{len(market_df)} registros de mercado carregados para o período de {start_year}-{end_year}.")

    # 2. Carregar e preparar os dados de opções
    try:
        options_df = pd.read_csv(options_csv_path)
        # Supondo que a coluna de tempo se chame 'time' ou 'timestamp'
        time_col = 'time' if 'time' in options_df.columns else 'timestamp'
        options_df[time_col] = pd.to_datetime(options_df[time_col]).dt.date
        logger.info(f"{len(options_df)} registros de opções carregados de {options_csv_path}.")
    except FileNotFoundError:
        logger.error(f"Arquivo de opções não encontrado em: {options_csv_path}")
        raise

    merged_rows = []
    for _, market_row in market_df.iterrows():
        current_date = market_row['timestamp']
        spot_price = market_row['close']

        daily_options = options_df[options_df[time_col] == current_date]

        if not daily_options.empty:
            # Lógica para encontrar a melhor opção: a mais próxima do dinheiro (at-the-money)
            # com o vencimento mais curto.
            best_option = daily_options.iloc[
                (daily_options['strike'] - spot_price).abs().argsort()
            ].iloc[0]

            # Unir as informações
            merged_row = market_row.copy()
            merged_row['strike'] = best_option['strike']
            merged_row['premium'] = best_option['premium']
            merged_row['days_to_maturity'] = best_option['days_to_maturity']
            # Garante que a volatilidade está em decimal
            merged_row['volatility'] = best_option['volatility'] / 100.0 if best_option['volatility'] > 1.0 else best_option['volatility']
            
            merged_rows.append(merged_row)

    final_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    logger.info(f"Fusão completa. O dataset final contém {len(final_df)} registros.")
    
    return final_df