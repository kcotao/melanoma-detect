# Base con PyTorch CPU ya instalado (evita compilar)
FROM pytorch/pytorch:2.4.0-cpu
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY weights ./weights

ENV MODEL_PATH=weights/resnet18_finetune_best.pth
ENV MODEL_VERSION=1.0.0

EXPOSE 7860

# Gradio queda montado en "/" y Uvicorn sirve FastAPI
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

