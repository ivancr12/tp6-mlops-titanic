FROM python:3.9-slim

WORKDIR /app

# Copiar todo el contenido del directorio actual (incluye train.csv)
COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p models && \
    python scripts/train_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
