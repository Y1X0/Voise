# Voise voice engine for a free Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM).
# Models are downloaded at BUILD time and baked into the image, so restarts are fast and nothing is fetched
# while serving a request.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchaudio==2.11.0 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps omnivoice==0.2.1

COPY fetch_models.py .
RUN python fetch_models.py /models && chown -R user /models

COPY app.py .
USER user
ENV MODELS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
