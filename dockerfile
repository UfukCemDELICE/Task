# Base image olarak Python 3.13 slim
FROM python:3.13-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları (bazı Python paketleri için gerekli)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Uygulama bağımlılıklarını yükle
RUN pip install --no-cache-dir \
    fastapi>=0.118.0 \
    httpx>=0.28.1 \
    huggingface-hub>=0.35.3 \
    ipykernel>=6.30.1 \
    matplotlib>=3.10.6 \
    numpy>=2.3.3 \
    pandas>=2.3.3 \
    peft>=0.17.1 \
    pydantic>=2.11.10 \
    pylint>=3.3.9 \
    pytest>=8.4.2 \
    scikit-learn>=1.7.2 \
    torch>=2.8.0 \
    transformers>=4.56.2 \
    uvicorn>=0.37.0

# Uygulama kodunu kopyala
COPY . /app

# UVicorn ile uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
