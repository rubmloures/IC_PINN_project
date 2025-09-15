# scripts/create_db.py
import sys
import os

# Adiciona a pasta raiz do projeto ao caminho de busca do Python
# 1. Pega o caminho do diretório do script atual (.../scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Sobe um nível para a raiz do projeto (.../trading_system)
project_root = os.path.dirname(script_dir)
# 3. Insere no início da lista de caminhos do sistema
sys.path.insert(0, project_root)
import logging
from trading_system.db.database_manager import db_manager

# Configuração básica de logging para ver a saída no console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_database():
    """Lê o arquivo schema.sql e executa os comandos para criar as tabelas."""
    try:
        with open('scripts/schema.sql', 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                logging.info("Executando script de inicialização do banco de dados (schema.sql)...")
                cur.execute(sql_script)
                conn.commit()
        logging.info("Banco de dados inicializado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao inicializar o banco de dados: {e}", exc_info=True)
    finally:
        db_manager.close_pool()

if __name__ == "__main__":
    initialize_database()