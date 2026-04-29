# Dockerfile для Python окружения (для будущего использования)
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements (создать когда будет Python код)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Копирование Python файлов
# COPY python/ ./python/
# COPY data/ ./data/

# По умолчанию запускаем Python скрипт
# CMD ["python", "python/main.py"]

