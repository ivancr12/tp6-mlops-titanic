FROM python:3.9-slim

WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY ./app ./app
COPY ./models ./models

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
