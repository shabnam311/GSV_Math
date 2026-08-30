FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools + Python
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip curl wget \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp from source (CPU-optimized with AVX2)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp \
    && cd /llama.cpp \
    && cmake -B build -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/llama-server \
    && cp build/bin/llama-mtmd-cli /usr/local/bin/llama-mtmd-cli 2>/dev/null || true \
    && rm -rf /llama.cpp

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY app.py /app/app.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create model directory
RUN mkdir -p /models

# HF Spaces expects port 7860
EXPOSE 7860

WORKDIR /app
CMD ["./start.sh"]
