FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости, необходимые для сборки chromadb, hnswlib, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Команда запуска
CMD ["python", "main.py"]
