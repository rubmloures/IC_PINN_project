import os
from dotenv import load_dotenv
from enum import Enum

# Carrega as variáveis de ambiente do arquivo.env
load_dotenv()

class OperatingMode(Enum):
    """Define os modos de operação para o robô."""
    LIVE = "LIVE"
    PAPER = "PAPER"

# --- Configurações Gerais ---
# Modo de operação: 'LIVE' para negociação real, 'PAPER' para simulação.
OPERATING_MODE_STR = os.getenv("OPERATING_MODE", "PAPER").upper()
try:
    OPERATING_MODE = OperatingMode[OPERATING_MODE_STR]
except KeyError:
    raise ValueError(f"Modo de operação inválido: {OPERATING_MODE_STR}. Deve ser 'LIVE' ou 'PAPER'.")

# --- Configurações do Banco de Dados ---
# Estas variáveis são lidas do arquivo.env
DB_HOST = "localhost" # Garante que o host do banco seja o serviço 'postgres' do docker-compose
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "data_trading")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Validação de todas as variáveis necessárias
missing_vars = []
for var, value in zip([
    "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "DB_HOST", "DB_PORT"],
    [DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
    if not value:
        missing_vars.append(var)
if missing_vars:
    raise ValueError(f"As seguintes variáveis de ambiente estão faltando no arquivo .env: {', '.join(missing_vars)}")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Configurações da API da Plataforma ---
PLATFORM_API_KEY = os.getenv("PLATFORM_API_KEY")
PLATFORM_API_SECRET = os.getenv("PLATFORM_API_SECRET")

# --- Configurações de Alerta (Telegram) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")