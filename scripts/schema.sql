-- scripts/schema.sql

-- Tabela para armazenar dados históricos de mercado para treinamento e backtesting.
-- Esta tabela será a fonte de dados para o ambiente do TensorTrade.
CREATE TABLE IF NOT EXISTS historical_market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (timestamp, symbol)
);

-- Tabela para registrar o contexto completo de cada ciclo de decisão.
-- Essencial para auditoria, análise de performance e retreinamento do modelo.
CREATE TABLE IF NOT EXISTS decision_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_asset VARCHAR(50) NOT NULL,
    final_action VARCHAR(10) NOT NULL CHECK (final_action IN ('BUY', 'SELL', 'HOLD')),
    decision_context JSONB NOT NULL -- Armazena dados de mercado e votos individuais das estratégias.
);

-- Tabela para registrar todas as ordens executadas (reais ou simuladas).
CREATE TABLE IF NOT EXISTS executed_orders (
    id SERIAL PRIMARY KEY,
    decision_log_id INTEGER REFERENCES decision_log(id), -- Chave estrangeira para ligar a ordem à decisão.
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    fees NUMERIC(20, 8) DEFAULT 0.0,
    order_status VARCHAR(20) NOT NULL,
    is_paper_trade BOOLEAN NOT NULL,
    platform_order_id VARCHAR(255) UNIQUE
);

-- Tabela para snapshots periódicos do valor do portfólio.
-- Usada para gerar gráficos de desempenho e monitorar a performance geral.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_value_usd NUMERIC(20, 8) NOT NULL,
    cash_usd NUMERIC(20, 8) NOT NULL,
    holdings JSONB -- Armazena a quantidade de cada ativo como um objeto JSON.
);

-- Índices para otimizar consultas em colunas frequentemente acessadas.
CREATE INDEX IF NOT EXISTS idx_historical_market_data_timestamp ON historical_market_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_log_timestamp ON decision_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_executed_orders_timestamp ON executed_orders(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp DESC);