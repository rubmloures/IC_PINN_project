# scripts/fetch_historical_data.py
import time
import logging
import certifi
import os
os.environ['SSL_CERT_FILE'] = r'D:\UERJ\Programação e Códicos\trading_system\cacert.pem'
import pandas as pd
import yfinance as yf
import psycopg_pool
import dotenv
from trading_system.db.database_manager import db_manager

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Defina os tickers e o período desejado
TICKERS = [
    "PETR4.SA",
    #"BBAS3.SA",
    #"ITUB4.SA",
    #"BBDC4.SA",
    #"ITSA4.SA",
    #"BRAP4.SA",
    #"VALE3.SA",
    #"USIM5.SA",
    #"CSNA3.SA",
    #"EMBR3.SA",
]  # Corrija para os tickers desejados
START_DATE = '2024-09-01'
END_DATE = '2025-09-01'

def fetch_and_store_data():
    """Baixa dados do yfinance e os armazena no banco de dados."""
    logging.info(f"Baixando dados para os tickers: {TICKERS}")
    
    all_data = [] # Lista para armazenar os dataframes de cada ticker

    try:
        for ticker in TICKERS:
            logging.info(f"Baixando dados para o ticker: {ticker}...")
            # Baixa os dados de UM ticker de cada vez
            data = yf.download(ticker, start=START_DATE, end=END_DATE)
            
            if data.empty:
                logging.warning(f"Nenhum dado baixado para {ticker}. Pulando.")
                continue

            # Adiciona a coluna 'symbol' pois agora não é um MultiIndex
            data['symbol'] = ticker
            all_data.append(data)

            # Pausa educada para não sobrecarregar o servidor
            time.sleep(10) # Pausa por 10 segundos

        if not all_data:
            logging.error("Nenhum dado foi baixado para nenhum ticker.")
            return

        # Concatena todos os dataframes da lista em um só
        df = pd.concat(all_data).reset_index()

        # Renomeia as colunas para o padrão do banco de dados
        df = df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Manter apenas as colunas necessárias na ordem correta
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        logging.info(f"Total de {len(df)} linhas de dados baixadas para {len(all_data)} tickers.")
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                logging.info("Inserindo dados na tabela historical_market_data...")
                # ... (o resto do código para inserir no banco de dados continua igual)
                insert_query = """
                    INSERT INTO historical_market_data (timestamp, symbol, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol) DO NOTHING;
                """
                data_to_insert = list(df.itertuples(index=False, name=None))
                cur.executemany(insert_query, data_to_insert)
                conn.commit()
        logging.info("Inserção de dados concluída.")

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a coleta de dados: {e}", exc_info=True)
    finally:
        db_manager.close_pool()

if __name__ == "__main__":
    fetch_and_store_data()