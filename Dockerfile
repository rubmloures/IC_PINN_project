# Base pequena e estável
FROM python:3.11-slim

# Boas práticas p/ Python em containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Só bibliotecas *de runtime* (Postgres + certificados)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Copia só requirements p/ cachear instalações
COPY requirements.txt .

# 2) Instala deps sem cache do pip
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 3) Agora copia o código (não invalida o cache das deps sem necessidade)
COPY . .

# Usuário sem root
RUN useradd -ms /bin/bash appuser
USER appuser

# Seu comando original
CMD ["python", "scripts/create_db.py"]