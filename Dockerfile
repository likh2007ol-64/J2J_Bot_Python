FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей (gcc/g++ нужны для некоторых Python-пакетов)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего кода
COPY . .

# Команда запуска
CMD ["python", "main.py"]
