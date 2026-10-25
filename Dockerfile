FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p \
    /workspace/data \
    /workspace/hf_cache

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
