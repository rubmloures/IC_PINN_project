import logging
import backoff
import requests
from.. import config

logger = logging.getLogger(__name__)

# Função de verificação para o decorador backoff.
# Desiste de retentar para erros de cliente (4xx), pois provavelmente não serão resolvidos com uma nova tentativa.
def is_client_error(e):
    """Retorna True se o status code for um erro 4xx, exceto 429 (Too Many Requests)."""
    if not isinstance(e, requests.exceptions.HTTPError):
        return False
    is_client_err = 400 <= e.response.status_code < 500
    is_rate_limit = e.response.status_code == 429
    return is_client_err and not is_rate_limit

# Decorador personalizado para encapsular a lógica de retentativa
on_transient_error = backoff.on_exception(
    backoff.expo,
    (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError),
    max_tries=8,
    max_time=300, # Tempo máximo total de 5 minutos
    giveup=is_client_error,
    logger=logger
)

class TradingPlatformAPI:
    """
    Isola toda a comunicação com a API da plataforma de trading.
    """
    def __init__(self):
        self.base_url = "https://api.targetplatform.com/v1" # URL de exemplo
        self.api_key = config.PLATFORM_API_KEY
        self.api_secret = config.PLATFORM_API_SECRET
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        })

    @on_transient_error
    def get_account_balance(self) -> dict:
        """Busca o balanço da conta."""
        logger.info("Buscando balanço da conta...")
        response = self.session.get(f"{self.base_url}/account/balance")
        response.raise_for_status() # Lança HTTPError para status codes 4xx/5xx
        balance_data = response.json()
        logger.info("Balanço da conta obtido com sucesso.")
        return balance_data

# Instância global para ser importada por outros módulos
api_connector = TradingPlatformAPI()