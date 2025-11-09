# Imagen base liviana
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Opcional) dependencias del sistema que a veces requiere torch/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Primero instala las deps de Python (sin torch/torchvision en requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch CPU y torchvision desde el índice oficial CPU
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1

# Copiar código y pesos
COPY src ./src
COPY app ./app
COPY weights ./weights

ENV MODEL_PATH=weights/resnet18_finetune_best.pth \
    MODEL_VERSION=1.0.0

EXPOSE 7860
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
