# Comando para rodar o script: python scripts/import_cotahist_data.py

# scripts/import_cotahist_data.py
import os
import pandas as pd
import logging
from trading_system.db.database_manager import db_manager
from datetime import datetime

# --- CONFIGURAÇÃO ---
COTAHIST_FOLDER_PATH = "data\Data_Spotprice_hist"
# Use a saída do debug_tickers.py atualizado para montar esta lista
TICKERS_TO_IMPORT = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", 
    "ITSA4", "BRAP4", "USIM5", "CSNA3", "EMBR3"
]
# --------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_cotahist_line(line):
    record_type = line[0:2]
    codbdi = line[10:12]

    # Retorna None se não for um registro de cotação do mercado à vista (BDI '02')
    if record_type != '01' or codbdi != '02':
        return None

    try:
        date_str = line[2:10]
        ticker = line[12:24].strip()
        open_price_str = line[56:69]
        high_price_str = line[69:82]
        low_price_str = line[82:95]
        close_price_str = line[108:121]
        volume_str = line[170:188]

        record = {
            'timestamp': datetime.strptime(date_str, '%Y%m%d'),
            'symbol': ticker,
            'open': float(open_price_str) / 100.0,
            'high': float(high_price_str) / 100.0,
            'low': float(low_price_str) / 100.0,
            'close': float(close_price_str) / 100.0,
            'volume': int(volume_str)
        }
        return record
    except (ValueError, IndexError) as e:
        logging.warning(f"Não foi possível processar a linha: {line[:30]}... Erro: {e}")
        return None

def import_cotahist_data():
    logging.info(f"Iniciando importação dos arquivos da pasta: {COTAHIST_FOLDER_PATH}")
    
    all_records = []
    
    try:
        files_to_process = [f for f in os.listdir(COTAHIST_FOLDER_PATH) if f.upper().startswith('COTAHIST') and f.upper().endswith('.TXT')]
        if not files_to_process:
            logging.error("Nenhum arquivo COTAHIST encontrado no diretório.")
            return

        logging.info(f"Arquivos encontrados: {len(files_to_process)}")

        for filename in sorted(files_to_process):
            logging.info(f"Processando arquivo: {filename}...")
            file_path = os.path.join(COTAHIST_FOLDER_PATH, filename)
            
            with open(file_path, 'r', encoding='latin-1') as f:
                for line in f:
                    parsed_record = parse_cotahist_line(line)
                    if parsed_record and parsed_record['symbol'] in TICKERS_TO_IMPORT:
                        all_records.append(parsed_record)

        if not all_records:
            logging.error("Nenhum dado encontrado para os tickers especificados. Verifique a lista de tickers e se eles existem nos arquivos com BDI '02'.")
            return

        df = pd.DataFrame(all_records)
        logging.info(f"Total de {len(df)} registros processados para {len(TICKERS_TO_IMPORT)} tickers.")
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                logging.info(f"Inserindo {len(df)} registros na tabela historical_market_data...")
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
        logging.error(f"Ocorreu um erro durante a importação: {e}", exc_info=True)
    finally:
        db_manager.close_pool()

if __name__ == "__main__":
    import_cotahist_data()