import logging
import threading
from contextlib import contextmanager
from psycopg_pool import ConnectionPool
from.. import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Gerencia um pool de conexões com o banco de dados PostgreSQL.
    Implementa o padrão Singleton para garantir uma única instância do pool.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            logger.info("Inicializando o DatabaseManager...")
            try:
                self.pool = ConnectionPool(
                    conninfo=config.DATABASE_URL,
                    min_size=2,
                    max_size=10,
                    timeout=30,
                    name="trading_bot_pool"
                )
                # Testa a conexão inicial para garantir que o pool foi criado com sucesso
                with self.pool.connection() as conn:
                    logger.info(f"Pool de conexões criado com sucesso. Versão do Servidor DB: {conn.pgconn.server_version}")
                self._initialized = True
            except Exception as e:
                logger.critical(f"Falha ao inicializar o pool de conexões do banco de dados: {e}", exc_info=True)
                raise

    @contextmanager
    def get_connection(self):
        """Fornece uma conexão do pool como um gerenciador de contexto."""
        try:
            with self.pool.connection() as conn:
                logger.debug("Conexão adquirida do pool.")
                yield conn
        except Exception as e:
            logger.error(f"Erro ao adquirir conexão do pool: {e}", exc_info=True)
            raise
        finally:
            logger.debug("Conexão devolvida ao pool.")
    
    def execute_query(self, query, params=None, fetch=None):
        """
        Executa uma query de forma segura com parâmetros.
        :param query: String SQL com placeholders (%s).
        :param params: Tupla de parâmetros para substituir na query.
        :param fetch: 'one', 'all', ou None para commit.
        :return: Resultado da query se fetch for especificado, senão None.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query, params)
                    if fetch == 'one':
                        return cur.fetchone()
                    if fetch == 'all':
                        return cur.fetchall()
                    conn.commit()
                except Exception as e:
                    logger.error(f"Query no banco de dados falhou. Revertendo. Erro: {e}", exc_info=True)
                    conn.rollback()
                    raise

    def close_pool(self):
        """Fecha todas as conexões no pool."""
        if hasattr(self, 'pool') and self.pool and not self.pool.closed:
            logger.info("Fechando o pool de conexões do banco de dados.")
            self.pool.close()

# Instância global para ser importada por outros módulos
db_manager = DatabaseManager()